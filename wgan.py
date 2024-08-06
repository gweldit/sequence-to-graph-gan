import argparse
import os
import random
from collections import Counter

import torch
import torch.nn as nn
from custom_dataset import CustomSequenceDataset, collate_fn
from file_reader import encode_sequences, load_and_print_dataset
from graph_encoder import GraphEncoder
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from wgan_gp import CustomSoftmax 

from helper_tools import get_device
from utils import set_random_seeds



class RelaxedEmbedding(nn.Module):
    """
    A drop-in replacement for `nn.Embedding` such that it can be used _both_ with Reinforce-based training
    and with Gumbel-Softmax one.
    Important: nn.Linear and nn.Embedding have different initialization strategies, hence replacing nn.Linear with
    `RelaxedEmbedding` might change results.
    """

    def __init__(self, vocab_size, embedding_dim):
        super(RelaxedEmbedding, self).__init__()
        self.device= get_device()
        self.embed_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, device=self.device)
        
    def forward(self, x):
        """this code is taken from github: 
        [source](https://github.com/facebookresearch/EGG/blob/170e5fe63c13244121a5b29b9bfb4870a0f11796/egg/core/gs_wrappers.py#L203)"""
        if isinstance(x, torch.LongTensor) or (x.dtype == torch.long) or(
            torch.cuda.is_available() and isinstance(x, torch.cuda.LongTensor)):

            return self.embed_layer(x)

        else:
            return torch.matmul(x, self.embed_layer.weight.to(self.device))
class Generator(nn.Module):
    def __init__(self, vocab_size, num_classes, num_layers, softmax='relaxed_bernoulli'):
        super(Generator, self).__init__()
        self.embedding_dim = 32
        self.hidden_dim = 64

        self.seq_len = 100
        self.latent_dim = 100
        self.vocab_size = vocab_size
        self.act = softmax
        self.label_embedding = nn.Embedding(num_classes, self.embedding_dim)
        self.noise_embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim * 2, self.hidden_dim, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_dim, vocab_size)

        self.custom_softmax = CustomSoftmax(self.act) 
    
    def forward(self, noise, labels, tau=0.2, anneal_tau=False, hard=False):
        label_embed = self.label_embedding(labels)
        noise_embed = self.noise_embedding(noise)
        combined = torch.cat((noise_embed, label_embed.unsqueeze(1).repeat(1, self.seq_len, 1)), -1)
        lstm_out, _ = self.lstm(combined)
        out = self.linear(lstm_out)
        # perform categorical sampling using custom softmax
        out = self.custom_softmax(out,tau=tau, hard=hard)
        return out
    


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.context_vector = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, hidden, lstm_outputs):
        """"""
        attn_scores = self.context_vector(torch.tanh(self.attn(lstm_outputs))).squeeze(-1)

        attn_weights = torch.softmax(attn_scores, dim=1)

        weighted_sum = torch.sum(hidden * attn_weights.unsqueeze(-1), dim=1)
        return weighted_sum, attn_weights


class Discriminator(nn.Module):
    def __init__(self, vocab_size,num_classes, num_layers, dropout=0.1, use_gpu=True):
        super(Discriminator, self).__init__()
        self.embedding_dim = 32
        self.hidden_dim = 64
        self.device = get_device() if use_gpu else torch.device('cpu')
        self.label_embedding = nn.Embedding(num_classes, self.embedding_dim, device=self.device)
        self.sequence_embedding = RelaxedEmbedding(vocab_size, self.embedding_dim).to(self.device)
        self.lstm = nn.LSTM(self.embedding_dim * 2, self.hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)

        self.attention  = Attention(self.hidden_dim)
        self.linear = nn.Linear(self.hidden_dim, 1)
        # self.sigmoid = nn.Sigmoid()
    
    def forward(self, seq, labels, lengths):
        label_embed = self.label_embedding(labels)
        
        seq_embed = self.sequence_embedding(seq)
        # print("sequence=",seq_embed[0])
        combined = torch.cat((seq_embed, label_embed.unsqueeze(1).repeat(1, seq.size(1), 1)), -1)
        packed_input = pack_padded_sequence(combined, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        attn_output, _ = self.attention(output, output)

        out = self.linear(attn_output)
        # out = self.linear(output[range(output.size(0)), lengths-1, :])  # Use the last valid output
        # out = self.sigmoid(out)
        # print("output of discriminator shape: ", out.shape)
        return out



class CWGAN(nn.Module):
    def __init__(self,generator,discriminator, opt):
        super(CWGAN, self).__init__()
        self.device = get_device()
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)
        
        self.d_optimizer = Adam(self.discriminator.parameters(),lr=opt.lr, betas= (opt.beta1, opt.beta2))
        
        self.g_optimizer = Adam(self.generator.parameters(),lr=opt.lr, betas= (opt.beta1, opt.beta2))

        self.clip_value = opt.clip_value
        self.critic_iterations = opt.n_critic

        self.losses = {"d_loss":[],"g_loss":[]}

        # self.loss_fn = self.wasserstein_loss


    def wasserstein_loss(self, D_real, D_fake):
        """computes wasserstein loss for the critic"""
        return - (torch.mean(D_real) - torch.mean(D_fake))
    
    def generator_loss(self,  D_fake):
        """computes generator loss"""
        return -torch.mean(D_fake)

    

    def clip_weights(self,clip_value):
        """applies weight clip to the discriminator"""
        for p in self.discriminator.parameters():
            p.data.clamp_(-clip_value, clip_value)



    def train_step(self, train_dataloader, tau, hard, ):
        d_loss_batch = 0.0
        g_loss_batch = 0.0
        number_of_batchs = len(train_dataloader)
        
        for i, (sequences, labels, lengths) in enumerate(tqdm(train_dataloader)):
            
            # -----------------------------
            # Train Disciriminator
            # -----------------------------
            # print("class_weights:", Counter(labels.tolist()))

            self.discriminator.zero_grad()
            real_data = sequences.to(self.device)
            # print("real data shape = ", real_data.shape)
            labels = labels.to(self.device)
            batch_size = labels.size(0)
            fake_lengths = torch.tensor([self.generator.latent_dim] * batch_size, dtype=torch.long)
            
            real_validity = self.discriminator(real_data, labels, lengths).to(self.device)

            noise  = torch.randint(0, self.generator.vocab_size, size=(labels.size(0), self.generator.latent_dim)).to(self.device)

            gen_data = self.generator(noise, labels, tau, hard).detach()
            # print("gen data: ", gen_data.shape)
            fake_validity = self.discriminator(gen_data, labels, fake_lengths)

            d_loss = self.wasserstein_loss(real_validity, fake_validity)

            d_loss.backward()
            self.d_optimizer.step()

            # update batch loss
            d_loss_batch += d_loss.item()

            # perform weight clipping
            self.clip_weights(clip_value=self.clip_value)
            
            # -------------------------------------
            # Train Generator
            # -------------------------------------

            # noise = torch.randint(0, self.generator.vocab_size, size=(labels.size(0), self.generator.latent_dim)).to(self.device)

            if i % self.critic_iterations:
                """updates generator at every n_critic iterations"""
                self.generator.zero_grad()
                noise = torch.randint(0, self.generator.vocab_size, size=(batch_size,self.generator.latent_dim)).to(self.device)
                gen_data = self.generator(noise, labels, tau, hard)

                fake_validity = self.discriminator(gen_data, labels, fake_lengths)

                g_loss = self.loss(fake_validity, torch.ones_like(fake_validity).to(self.device))

                g_loss.backward()
                self.g_optimizer.step()

                # update batch loss
                g_loss_batch += g_loss.item()

        # update epoch loss
        d_loss_batch /= number_of_batchs
        g_loss_batch /= number_of_batchs
        self.losses["d_loss"].append(d_loss_batch)
        self.losses["g_loss"].append(g_loss_batch)

        return d_loss_batch, g_loss_batch

    def train(self, train_dataloader, epochs, tau, hard=False, anneal_tau=False, anneal_interval=5):
        
        initial_tau = 5.0 if anneal_tau else tau
        final_tau = 0.1 if anneal_tau else tau

        for epoch in range(epochs):
            d_loss, g_loss = self.train_step(train_dataloader, tau=initial_tau, hard=hard)
            print(f"Epoch {epoch + 1} / {epochs}| Discriminator Loss: {d_loss:.5f} | Generator Loss: {g_loss:.5f}")

            if anneal_tau:
                initial_tau = CustomSoftmax.anneal_tau_(initial_tau, final_tau, epoch, epochs, anneal_interval)
                print("epoch = ", epoch + 1, "| tau = ", initial_tau)
        

    def validate(self, val_dataloader):
        # generate fake data and evaluate the error using Wasserstein distance or MSELoss or visual image graph
        pass



def sample_equal_data(train_data, train_labels):
    sampled_train_data = []
    sampled_train_labels = []

    # Combine data and labels
    train_data_with_labels = list(zip(train_data, train_labels))
    # Shuffle data
    random.shuffle(train_data_with_labels)

    # Count classes
    class_counts = Counter(train_labels)
    min_class_count = min(class_counts.values())

    # Initialize a dictionary to keep track of sampled data for each class
    sampled_data_dict = {label: [] for label in class_counts.keys()}

    # Sample data
    for data, label in train_data_with_labels:
        if len(sampled_data_dict[label]) < min_class_count:
            sampled_data_dict[label].append(data)

    # Flatten the sampled data and labels
    for label, data_list in sampled_data_dict.items():
        sampled_train_data.extend(data_list)
        sampled_train_labels.extend([label] * len(data_list))

    return sampled_train_data, sampled_train_labels



def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs",type=int, default = 50,)
    parser.add_argument("--batch_size",type=int, default = 64,)
    parser.add_argument("--num_classes",type=int, default = 2,)
    parser.add_argument("--vocab_size",type=int, default = 175,)
    parser.add_argument("--seq_length",type=int, default = 100,)
    parser.add_argument("--num_layers",type=int, default = 2,)
    parser.add_argument("--dropout",type=float, default = 0.3,)
    parser.add_argument('--softmax', type=str, default='gumbel_softmax')
    parser.add_argument("--hard",type=bool, default = False,)
    parser.add_argument("--tau",type=float, default = 0.2,)
    parser.add_argument("--anneal_tau",type=bool, default = False,)
    parser.add_argument("--anneal_interval",type=int, default = 5)
    parser.add_argument("--lr",type=float,default = 0.0002,)
    parser.add_argument("--beta1",type=float, default = 0.5,)
    parser.add_argument("--beta2",type=float, default = 0.999,)
    parser.add_argument("--clip_value",type=float, default = 0.1)
    parser.add_argument('--n_critic', type=int, default=5)

    opt = parser.parse_args()

    generator = Generator(vocab_size=opt.vocab_size, num_classes=opt.num_classes, num_layers=opt.num_layers + 1, softmax=opt.softmax)
    discriminator = Discriminator(vocab_size=opt.vocab_size, num_classes=opt.num_classes, num_layers=opt.num_layers, dropout=opt.dropout)
    device = get_device()
    model = CWGAN(generator, discriminator, opt).to(device)

    # generate fake dataset
    

    # data = torch.randint(0, opt.vocab_size,size=(opt.batch_size, opt.seq_length))
    # labels = torch.randint(0, opt.num_classes,size=(opt.batch_size,))

    # dataset = CustomSequenceDataset(data, labels)

    # train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn)

    # model.train(train_dataloader, epochs=opt.epochs, tau=0.2, hard=True)


    set_random_seeds(seed_value=42)
    

    # data_folder_path = "ADFA"
    vocabs = GraphEncoder.load_vocabulary()
    dataset_folder = os.path.join(os.getcwd(), "data")

    train_data, train_labels = load_and_print_dataset(os.path.join(dataset_folder, "train_dataset.json"),print_data=False)

    test_data, test_labels = load_and_print_dataset(os.path.join(dataset_folder, "test_dataset.json"),print_data=False)

    # encode sequences using vocabulary
    train_data = encode_sequences(train_data, vocabs)
    test_data = encode_sequences(test_data, vocabs)

    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_labels)
    test_labels = label_encoder.transform(test_labels)
    # print(collections.Counter(test_labels))


    max_length = opt.seq_length # Controls the sequence length of inputs to Discriminator. (LSTM)
    batch_size = opt.batch_size 

    # sample equal class  distribution in the train data
    train_data, train_labels = sample_equal_data(train_data, train_labels)

    train_dataset = CustomSequenceDataset(train_data, train_labels, length=max_length)
    test_dataset = CustomSequenceDataset(test_data, test_labels, max_length)

    # perform balancing sampling
    # class_weights = calculate_class_weights((train_labels.tolist()))
    # class_weights = torch.tensor([1.0, 1.2])
    # print("class_weights:", class_weights)

    print("labels count =", Counter(train_labels))


    # sampler = torch.utils.data.WeightedRandomSampler(
    #     class_weights, num_samples=len(train_dataset.labels),)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, sampler=sampler)

    val_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    model.train(train_loader, epochs=opt.epochs, tau=opt.tau, hard=opt.hard,anneal_tau=opt.anneal_tau, anneal_interval=opt.anneal_interval)
    model.validate(val_loader)

    # save the generator

    torch.save(model.generator.state_dict(), 'saved_models/wgan_generator.pt')
    torch.save(model.discriminator.state_dict(), 'saved_models/wgan_discriminator.pt')




if __name__ == "__main__":
    main() 