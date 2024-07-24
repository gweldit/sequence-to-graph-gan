
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config import get_args
from custom_dataset import CustomSequenceDataset, collate_fn
from preprocess_data import load_sequence_dataset
from sklearn.metrics import f1_score
from torch import autograd, mean, ones, rand, randn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from helper_tools import get_device, set_random_seeds


class CustomSoftmax(nn.Module):
    """Implements gumbel softmax and relaxed bernoulli (softmax with temperature) to perform categorical sampling."""

    def __init__(self, act="gumbel_softmax"):
        """
        Initializes the CustomSoftmax module.

        Parameters:
            act (str): The activation function to use. Defaults to "gumbel_softmax".
        """
        super().__init__()
        self.act = act

    def forward(self, logits, tau, hard):
        """
        Computes the forward pass of the model.

        Args:
            logits (torch.Tensor): The input logits.
            tau (float): The temperature parameter for the gumbel softmax, relaxed bernoulli, or normal softmax to perform categorical sampling. Defaults to normal softmax.
            hard (bool): Whether to perform hard (one-hot) or soft sampling.

        Returns:
            torch.Tensor: The output tensor after applying the specified activation function.

        """
        if self.act == "gumbel_softmax":
            return self.gumbel_softmax(logits, tau, hard)
        elif self.act == "relaxed_bernoulli":
            return self.relaxed_bernoulli(logits, tau)
        return F.softmax(logits, dim=-1) # default sampling

    def sample_gumbel(self, logits, eps=1e-10):
        """Sample from Gumbel(0, 1)"""
        U = eps + (1 - eps) * torch.rand(logits.size()).to(logits.device)
        # U = torch.rand_like(logits).to(logits.device)

        return -torch.log(-torch.log(U + 1e-20) + 1e-20)

    def gumbel_softmax_sample(self, logits, tau):
        """Draw a sample from the Gumbel-Softmax distribution"""
        y = logits + self.sample_gumbel(logits)
        return F.softmax(y / tau, dim=-1)

    def gumbel_softmax(self, logits, tau, hard=False):
        """Sample from the Gumbel-Softmax distribution and optionally discretize.
        Args:
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
        Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probability distribution that sums to 1 across classes
        """
        y = self.gumbel_softmax_sample(logits, tau)
        if hard:
            y_hard = torch.nn.functional.one_hot(
                y.argmax(dim=-1), num_classes=logits.size(-1)
            ).float()
            y = (y_hard - y).detach() + y

        return y

    def relaxed_bernoulli(self, logits, tau=1, dim=-1):
        """It is also known as a softmax with temperature, or relaxed softmax.
        torch.distributions.relaxed_bernoulli.RelaxedBernoulli(tau, probs=None, logits=logits, validate_args=None)

        Args:
            logits (tensors): tensors of [batch_size, L, vocab_size]
            tau (float): a temperature parameter to relax the distribution of logits. When tau = 1, it becomes a softmax.

        """

        return F.softmax(logits / tau, dim)

class Generator(nn.Module):
    def __init__(self, vocab_size, n_classes=2):
        super(Generator, self).__init__()
        self.latent_dim = 100
        self.seq_len = 120
        self.vocab_size = vocab_size
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            # *block(512, 1024),
            nn.Linear(512, self.vocab_size * self.seq_len)
        )

        self.label_embedding = nn.Embedding(n_classes, self.latent_dim)
        self.gumbel_softmax = CustomSoftmax()

    def forward(self, noise, labels, tau, hard):
        condition = self.label_embedding(labels)
        out = torch.mul(noise, condition)
        gen_data = self.model(out)
        gen_data = gen_data.view(noise.shape[0], self.seq_len, self.vocab_size)
        gen_data = self.gumbel_softmax(gen_data, tau=tau, hard=hard)
        return gen_data



class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, drop_rate=0.3):
        """
        Initializes the Discriminator.

        Args:
            input_size (int): The number of expected features in the input x.
            hidden_size (int): The number of features in the hidden state h.
            num_layers (int): Number of recurrent layers.
            num_classes (int): Number of classes or categories.
        """
        super(Discriminator, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=drop_rate)

        # n_classes + 1 (for real / fake classes)
        self.fc = nn.Linear(self.hidden_size, num_classes + 1) 
    
    def forward(self, packed_input, lengths=None):
        """
        Forward pass of the Discriminator.

        Args:
            packed_input (torch.Tensor): The input tensor in a packed sequence format.
            lengths (torch.Tensor, optional): The lengths of each sequence in the packed input. Defaults to None.

        Returns:
            torch.Tensor: The output tensor after applying the fully connected layer.

        Description:
            This function takes in a packed input tensor and passes it through an LSTM layer. 
            It then uses the final hidden state's output to classify the input. 
            The output tensor is obtained by applying a fully connected layer to the final hidden state.

        Note:
            - The input tensor should be in a packed sequence format.
            - The lengths tensor is optional and is used to unpack the packed sequence if provided.
            - The output tensor has the same shape as the final hidden state's output.

        """
        
        packed_output, (h_t, c_t) = self.lstm(packed_input)

        # use the final hidden state's output for classification
        output = h_t[-1]

        out = self.fc(output)
        return out

def compute_gradient_penalty(discriminator, encoded_padded_real_data, fake_data, labels, lengths):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = rand((fake_data.size(0), 1, 1)).to(fake_data.device)
    # Get random interpolation between real and fake samples

    padding_size = (encoded_padded_real_data.size(1) - fake_data.size(1))
    
    padded_fake_data = F.pad(fake_data, (0,0,0, padding_size))

    # print('shape of fake data = ', padded_fake_data.shape)

    # print("shape of real samples =", (encoded_padded_real_data.shape))
    # Pack the padded sequence
    interpolates = (alpha * encoded_padded_real_data + ((1 - alpha) * padded_fake_data)).requires_grad_(True)

    packed_input = pack_padded_sequence(interpolates, lengths, batch_first=True, enforce_sorted=False)
    
    # take only the adversarial outputs
    d_interpolates = discriminator(packed_input, lengths=lengths)[:,0] 
    fake = ones(fake_data.size(0), requires_grad=False).float().to(fake_data.device)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty



def train_wgan(generator, discriminator, optimizer_G, optimizer_D, train_dataloader, val_loader, opt, device):
    # ----------
    #  Training
    # ----------

    batches_done = 0
    weight = torch.tensor([1.0, 2.0], dtype=torch.float32,device=device)
    criterion = nn.CrossEntropyLoss(weight)

    for epoch in range(1, opt.epochs + 1):
        # train mode
        generator.train()
        discriminator.train()
        for i, (padded_sequences, labels, lengths) in enumerate(tqdm(train_dataloader)):

            # one-hot encode real data
            real_data = F.one_hot(padded_sequences, opt.vocab_size).float().to(device)

            packed_input = pack_padded_sequence(real_data, lengths, batch_first=True, enforce_sorted=False)


            labels = labels.to(device)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = randn(real_data.shape[0], generator.latent_dim).to(device)

            # Generate a batch of images
            fake_data = generator(z, labels, tau=opt.tau, hard=opt.hard) #.detach()

            # Real data
            discriminator_real_data_output = discriminator(packed_input, labels)
            real_validity = discriminator_real_data_output[:, 0]
            real_class_output = discriminator_real_data_output[:, 1:]

            # Fake data
            discriminator_fake_data_output = discriminator(fake_data)
            fake_validity = discriminator_fake_data_output[:,0]
            fake_class_output = discriminator_fake_data_output[:, 1:]

            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(discriminator, real_data, fake_data, labels, lengths)
            
            # compute classes loss
            d_class_loss = 0.5 * (criterion(real_class_output, labels) + criterion(fake_class_output, labels)) 


            # Adversarial loss
            d_loss = -mean(real_validity) + mean(fake_validity) + opt.lambda_gp * gradient_penalty

            d_loss = d_loss + d_class_loss

            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()

            # Train the generator every n_critic steps
            if i % opt.n_critic == 0:

                # -----------------
                #  Train Generator
                # -----------------

                # Generate a batch of images
                fake_data= generator(z, labels, tau=opt.tau, hard=opt.hard)
                # Loss measures generator's ability to fool the discriminator
                # Train on fake data
                discriminator_fake_data_output = discriminator(fake_data)
                fake_validity = discriminator_fake_data_output[:, 0]
                fake_class_output = discriminator_fake_data_output[:,1:]

                # compute g's classes loss
                g_class_loss = 0.5 * criterion(fake_class_output, labels)
                # compute adversarial loss
                g_loss = -mean(fake_validity) + 0.5 * g_class_loss

                g_loss.backward()
                optimizer_G.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, opt.epochs, i, len(train_dataloader), d_loss.item(), g_loss.item())
                )

                if epoch % 10 == 0:
                    # save the generator weights
                    torch.save(generator.state_dict(), f"wgan_generator_{epoch}.pth")                

                
                if epoch % 5 == 0: # validate every 5epochs
                    with torch.no_grad():
                        # set discriminator to eval mode
                        discriminator.eval()
                        f1_scores = []

                        for i, (padded_sequences, labels, lengths) in enumerate(val_loader):

                            # one-hot encode real data
                            real_data = F.one_hot(padded_sequences, opt.vocab_size).float().to(device)

                            packed_input = pack_padded_sequence(real_data, lengths, batch_first=True, enforce_sorted=False)

                            # Real data
                            discriminator_real_data_output = discriminator(packed_input, lengths)
                            real_validity = discriminator_real_data_output[:, 0]
                            real_class_output = discriminator_real_data_output[:, 1:]

                            #compute f1 score
                            preds = torch.argmax(real_class_output, dim=1).detach().cpu().tolist()
                            labels = labels.cpu().tolist()
                            f1_scores.append(f1_score(preds, labels))

                        avg_f1_score = torch.mean(torch.tensor(f1_scores, dtype=torch.float32))

                        print(f"f1 score: {avg_f1_score:.4f}")

                        # set discriminator back to train mode
                        discriminator.train()


def main():
    # get arguments
    opt = get_args()

    vocab_size = 343
    input_size = vocab_size  
    hidden_size = 50
    num_layers = 2
    num_classes = 2
    drop_rate = 0.1
    device = get_device()

    discriminator = Discriminator(input_size,hidden_size,num_layers, num_classes, drop_rate=drop_rate).to(device)
    generator = Generator(opt.vocab_size, num_classes).to(device)


    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


    set_random_seeds(seed_value=42)
    

    # data_folder_path = "ADFA"
    dataset_folder = os.path.join(os.getcwd(), "data")

    # train_sequences, train_labels = fetch_sequence_data(os.path.join(dataset_folder, "train_dataset.json"))
    train_data, train_labels = load_sequence_dataset(os.path.join(dataset_folder, "train_dataset.json"))

    test_data, test_labels = load_sequence_dataset(os.path.join(dataset_folder, "test_dataset.json"))

    # train and test plit       
    # train_data, test_data, train_labels, test_labels = train_test_split(
    #     sequences, labels, random_state=42, test_size=0.2, stratify=labels, shuffle=True
    # )

    max_length = 256 # change it to None for full sequence length
    batch_size = 128

    train_dataset = CustomSequenceDataset(train_data, train_labels, length=max_length)
    test_dataset = CustomSequenceDataset(test_data, test_labels, max_length)

    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

    val_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    train_wgan(generator, discriminator, optimizer_G, optimizer_D, train_loader, val_loader, opt, device)



if __name__ == "__main__":
    main()



