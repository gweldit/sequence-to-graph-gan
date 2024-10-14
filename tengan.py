# import argparse
import csv
import json
import os

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.classification import BinaryF1Score
from tqdm import tqdm

# from data_iter import GenDataModule, DisDataModule
from transformer import (CustomTransformerEncoderLayer, PositionalEncoding,
                         create_causal_mask, create_padding_mask)


class Generator(L.LightningModule):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers,dim_feedforward, seq_length, dropout=0.5, batch_first=False, save_dir='pretrain_models_checkpoints'):
        super(Generator, self).__init__()
        self.save_hyperparameters()
        self.embed_size = embedding_dim
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.max_seq_length= self.seq_length # 4000
        self.position_encoder = PositionalEncoding(self.embed_size, self.max_seq_length)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.transformer_encoder_layer = CustomTransformerEncoderLayer(self.embed_size,num_heads,dim_feedforward, dropout, batch_first)
        # self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer,num_layers, norm=None, enable_nested_tensor=True)
        self.fc = nn.Linear(self.embed_size, vocab_size)

        self.start_token = 1
        self.end_token = 2
        self.save_dir = save_dir
        

    def forward(self, input_tokens):
        
        # token embedding
        embedded = self.embedding(input_tokens)

        # add positional encoding
        embedded = self.position_encoder(embedded)

        # transformer encoder
        padding_mask = create_padding_mask(input_tokens)
        attn_mask = create_causal_mask(input_tokens.size(1))

        encoder_output = self.transformer_encoder(embedded, mask=attn_mask, src_key_padding_mask=padding_mask)
        output = self.fc(encoder_output)
        return output
    
    def step(self, batch, batch_idx):
        # loss_fn = nn.CrossEntropyLoss()
        # add end token to 
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        
        outputs = self.forward(inputs)
        
        # Make tensors contiguous and then flatten using view
        outputs = outputs.contiguous().view(-1, outputs.size(-1))  # Shape: (N * seq_length, vocab_size)
        targets = targets.contiguous().view(-1)  # Shape: (N * seq_length)
        
        # print("shape of output: ", outputs.shape)
        # print("shape of targets: ", targets.shape)


        loss = F.cross_entropy(outputs, targets)

        # Apply the padding mask to the loss
        # padding_mask = (targets != 0).float()
        # loss = F.cross_entropy(outputs, targets, reduction='none')
        # # Apply the padding mask to the loss
        # loss = loss * padding_mask.reshape(-1)
        # # Normalize the masked loss
        # loss = loss.sum() / padding_mask.sum()

        return loss
    
    # def step(self, batch, batch_idx):
    #     # gen_output = self.forward(batch)

    #     # # return loss of generator

    #     loss = self.pretrain_step(batch, batch_idx)

    #     return loss


    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log('g_train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # print(f"Train Step {batch_idx} - G Loss: {loss.item():.4f}")
        return loss 
    def validation_step(self, batch, batch_idx):
        self.eval()
        loss = self.step(batch, batch_idx)
        self.log('g_val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # print(f"Validation Step {batch_idx} - G Loss: {loss.item():.4f}")
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]

    def save_model(self):
        os.makedirs(self.save_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(self.save_dir, 'generator.ckpt'))

    def load_model(self, checkpoint_path):
        self.load_state_dict(torch.load(checkpoint_path, map_location=self.device))

class Discriminator(L.LightningModule):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers,dim_feedforward, seq_length, dropout=0.5, batch_first=False, save_dir='discriminator_checkpoints', use_wgan=False):
        super(Discriminator, self).__init__()
        self.embed_size = embedding_dim
        self.seq_length = seq_length
        self.f1_metric = BinaryF1Score() 


        # embedding tokens
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # position embedding
        self.position_encoder = PositionalEncoding(self.embed_size, self.seq_length)
        # transform encoder layer
        self.transformer_encoder_layer = CustomTransformerEncoderLayer(self.embed_size,num_heads,dim_feedforward, dropout, batch_first)
        
        # transformer's encoder object with n encoders
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer,num_layers, norm=None, enable_nested_tensor=True)
        self.linear = nn.Linear(self.embed_size, 1)
        # self.clf = nn.Linear(self.embed_size, 2) # two classes: real or fake
        # self.sigmoid = nn.Sigmoid()

        self.save_dir = save_dir
        # criterion
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.use_wgan = use_wgan

    def forward(self, tokens):
        # Step 1: Embedding and Positional Encoding
        embedded = self.embedding(tokens)
        embedded = self.position_encoder(embedded)
        
        # Step 2: Transformer Encoder
        padding_mask = create_padding_mask(tokens)
        attn_mask = create_causal_mask(tokens.size(1))
        encoder_output = self.transformer_encoder(embedded, mask=attn_mask, src_key_padding_mask=padding_mask)
        
        # Step 3: Apply Masking Before Mean Pooling
        mask = padding_mask.unsqueeze(-1).expand(encoder_output.size()).float()
        masked_encoder_output = encoder_output * mask
        
        # Step 4: Compute the Sum and Count of Non-Padding Tokens
        sum_output = masked_encoder_output.sum(dim=1)
        count_non_padding = mask.sum(dim=1)
        
        # Avoid division by zero
        count_non_padding = torch.clamp(count_non_padding, min=1e-9)
        
        # Step 5: Compute the Mean
        pooled_output = sum_output / count_non_padding
        
        # Step 6: Pass Through the Linear Layer
        output = self.linear(pooled_output)
        
        return output
    
    def step(self, batch, batch_idx):
        # real_data, fake_data = batch
        data, labels = batch
        preds = self(data) # [batch_size, embed_dim] 
        loss = torch.zeros(1)
        if self.use_wgan:
            d_real_loss = preds[labels == 1].mean()
            d_fake_loss = preds[labels == 0].mean()
            # fake_output = self(fake_data) 
            # Wasserstein GAN Loss: Critic maximizes E[D(x)] - E[D(G(z))]
            loss = d_fake_loss - d_real_loss
        else:
        
            loss = self.bce_loss(preds, labels.unsqueeze(1).float()) 

        # compute f1 score
        
        f1_score = self.f1_metric(torch.sigmoid(preds).squeeze(1), labels)  
        
        return loss, f1_score

    def training_step(self, batch, batch_idx):
        # if self.pretrain_discriminator:
        loss, f1_score = self.step(batch, batch_idx)
        self.log('d_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        self.log('train_f1_score', f1_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)    

        # print(f"Train Step {batch_idx} - Loss: {loss.item():.4f}")
        # print(f"Train Step {batch_idx} - F1 Score: {f1_score.item():.4f}")  
        return loss
    
    def validation_step(self, batch, batch_idx):
        # self.eval()
        loss, f1_score = self.step(batch, batch_idx)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        self.log('val_f1_score', f1_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # print(f"Validation Step {batch_idx} - Loss: {loss.item():.4f}")
        # print(f"Validation Step {batch_idx} - F1 Score: {f1_score.item():.4f}")

        return loss
    
    def test_step(self, batch, batch_idx):  
        loss = self.step(batch, batch_idx)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        print(f"Test Step {batch_idx} - Loss: {loss.item():.4f}")
        return loss

        # else:
        #     return self.discriminator_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0002)

    def save_model(self):
        os.makedirs(self.save_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(self.save_dir, 'discriminator.ckpt'))

    def load_model(self, checkpoint_path):
        self.load_state_dict(torch.load(checkpoint_path, map_location=self.device))





class SequenceGenerator:
    def __init__(self, model, tokenizer, max_len, batch_size):
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.batch_size = batch_size

    def sample(self):
        self.model.eval()
        finished = [False] * self.batch_size
        sample_tensor = torch.zeros((self.batch_size, self.max_len), dtype=torch.long).to(self.model.device)

        # Start with start token
        sample_tensor[:, 0] = self.tokenizer.token_to_int[self.tokenizer.start]

        with torch.no_grad():
            for i in range(1, self.max_len):
                # print(f"Sampling step {i}/{self.max_len - 1}")

                logits  = self.model(sample_tensor[:, :i])
                probabilities = F.softmax(logits[:, -1, :], dim=-1)
                sampled_token = torch.multinomial(probabilities, 1).squeeze()

                for idx in range(self.batch_size):
                    if finished[idx]:
                        sampled_token[idx] = self.tokenizer.token_to_int[self.tokenizer.end]
                    if sampled_token[idx] == self.tokenizer.token_to_int[self.tokenizer.end]:
                        finished[idx] = True

                sample_tensor[:, i] = sampled_token

                if all(finished):
                    # print("All sequences finished.")
                    break

        self.model.train()
        return sample_tensor
    

    # def sample_multi(self, n, filename=None):
    #     samples = []
    #     for _ in tqdm(range(int(n / self.batch_size))):
    #         # Generate n batches
    #         print("generating batches ...")
    #         batch_sample = self.sample()
    #         samples.extend(batch_sample.detach().cpu().numpy())
    #     # Write the "n" samples into file
    #     if filename:
    #         with open(filename, 'w') as fout:
    #             for s in samples:
    #                 print("writing sample to a file ...")
    #                 fout.write('{}\n'.format(s))
    #     return samples
    

    def sample_multi(self, n, filename=None):
        samples = []
        for _ in tqdm(range(int(n / self.batch_size))):
            # Generate n batches
            print("generating batches ...")
            batch_sample = self.sample()
            samples.append(batch_sample.detach().cpu().numpy())

        samples = np.concatenate(samples, axis=0).tolist()

        # Write the "n" samples into file
        if filename:
            print("writing samples to a file ...")
            # df = pd.DataFrame(samples)
            # Write to CSV file
            with open('output.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(samples)


        return samples


class TenGAN(L.LightningModule):
    def __init__(self, generator, disscriminator, sequence_sampler, args):
        super(TenGAN, self).__init__()
        # self.generator = Generator(vocab_size=args.vocab_size,embedding_dim=args.embedding_dim,num_heads=args.num_heads,num_layers=args.num_encoders,dim_feedforward=args.dim_feedforward, seq_length=args.seq_length,dropout=args.dropout)

        # self.discriminator = Discriminator(vocab_size=args.vocab_size,embedding_dim=args.embedding_dim,num_heads=args.num_heads,num_layers=args.num_encoders,dim_feedforward=args.dim_feedforward, seq_length=args.seq_length,dropout=args.dropout)

        self.generator = generator
        self.discriminator = disscriminator
        self.sampler = sequence_sampler # use the same generator instance for sampling token-by-token sequences to compute RL-policy gradient

        self.criterion = nn.CrossEntropyLoss() # minimizing criterion is same as MLE
        self.bce_loss = torch.nn.BCEWithLogitsLoss()

        # flags
        # self.pre_training_gen = args.pre_training_gen
        # self.pre_training_disc = args.pre_training_disc
        self.adversarial_training = args.adversarial_training
        self.g_steps = args.g_steps
        self.use_wgan = args.use_wgan

        self.automatic_optimization = False

    # def forward(self, noise):
    #     return self.generator(noise)
    
    
    def compute_rewards(self, generated_sequences):
        with torch.no_grad():
            rewards = self.discriminator(generated_sequences).squeeze()
        return rewards

    def gen_step(self, generated_sequences, rewards):
        logits = self.generator(generated_sequences[:, :-1])
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(2, generated_sequences[:, 1:].unsqueeze(-1)).squeeze(-1)
        loss = -torch.mean(selected_log_probs * rewards.unsqueeze(-1))
        return loss

    def disc_step(self, real_sequences, generated_sequences):
        real_labels = torch.ones(real_sequences.size(0), 1).to(real_sequences.device)
        fake_labels = torch.zeros(generated_sequences.size(0), 1).to(generated_sequences.device)

        real_outputs = self.discriminator(real_sequences)
        fake_outputs = self.discriminator(generated_sequences)

        real_loss = F.binary_cross_entropy(real_outputs, real_labels)
        fake_loss = F.binary_cross_entropy(fake_outputs, fake_labels)

        loss = real_loss + fake_loss
        return loss

    def training_step(self, batch, batch_idx):
        real_sequences = batch
        # Access optimizers
        gen_optimizer, disc_optimizer = self.optimizers()

        # print("optimizer is instance of ", type(gen_optimizer))
        # Access current epoch
        
        print(f"Running Epoch: {self.current_epoch} / {self.trainer.max_epochs}")
        # Update generator for g_steps times
        gen_loss_total = torch.zeros(1).to(self.device)
        for _ in range(self.g_steps):
            # Generate sequences
            # print("Generating sequences ...")
            generated_sequences = self.sampler.sample()
            # print("Generating sequences is complete.")
            # Compute rewards
            rewards = self.compute_rewards(generated_sequences)
            # print("Generating rewards is complete")
            # Update generator
            gen_loss = self.gen_step(generated_sequences, rewards)
            gen_loss_total += gen_loss
            # print("gen loss is complete")
            # Zero gradients and perform backward pass for generator
            gen_optimizer.zero_grad()
            self.manual_backward(gen_loss)
            gen_optimizer.step()

            
        
        gen_loss_avg = gen_loss_total / self.g_steps
        # log losses
        self.log('gen_loss', gen_loss_avg.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        print("g loss = ", gen_loss_avg.item())
        
            
        

        
        # Generate sequences for discriminator update
        disc_loss = self.disc_step(real_sequences, generated_sequences)

        # Zero gradients and perform backward pass for discriminator
        disc_optimizer.zero_grad()
        self.manual_backward(disc_loss)
        disc_optimizer.step()
        # log loss
        self.log('disc_loss', disc_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        
        # print("d loss = ", disc_loss.item())
        
        # print(type(gen_loss_avg),"|", type(disc_loss))
        # return gen_loss_avg, disc_loss

    
        # # Log losses
        # self.log('gen_loss', gen_loss_avg)
        # self.log('disc_loss', disc_loss)

        # return gen_loss_avg,  disc_loss
    

    # def discriminator_step(self, real_data, fake_data):
    #     real_output = self.discriminator(real_data)
    #     fake_output = self.discriminator(fake_data.argmax(dim=-1))
    #     d_loss = -torch.mean(torch.log(real_output) + torch.log(1. - fake_output))
    #     return d_loss

    # def training_step(self, batch, batch_idx, optimizer_idx):
    #     real_data = batch
    #     if self.pre_training_gen:
    #         self.pretrain_generator(batch)

    #     if self.pretrain_discriminator:
    #         self.pretrain_discriminator(batch)

        
    #     # adversarial training
    #     if self.adversarial_training:
    #         noise = torch.randint(0, self.vocab_size, (batch.size(0), self.seq_length), device=self.device)
    #         fake_data = self.generator(noise)

    #         if optimizer_idx == 0:
    #             g_loss = self.generator_step(batch)
    #             self.log('g_loss', g_loss)
    #             return g_loss

    #         if optimizer_idx == 1:
    #             d_loss = self.discriminator_step(real_data, fake_data)
    #             self.log('d_loss', d_loss)
    #             return d_loss

    # def configure_optimizers(self):
    #     g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002)
    #     d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002)
    #     return [g_optimizer, d_optimizer], [] 


    def on_train_end(self):
        # Save the generator and discriminator models
        torch.save(self.generator.state_dict(), 'saved_models/adv_generator.pth')
        torch.save(self.discriminator.state_dict(), 'saved_models/adv_discriminator.pth')
        print("Models saved successfully.")
    
    def configure_optimizers(self):
        g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002)
        d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002)
        
        g_scheduler = optim.lr_scheduler.StepLR(g_optimizer, step_size=10, gamma=0.1)
        d_scheduler = optim.lr_scheduler.StepLR(d_optimizer, step_size=10, gamma=0.1)
        
        return [g_optimizer, d_optimizer] # , [g_scheduler, d_scheduler]
    def pretrain_generator(self, batch, batch_idx):
        pass
        

    def pretrain_discriminator(self, dataloader, epochs=1, use_wgan=True):
        optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002)
        for epoch in range(epochs):
            for batch in dataloader:
                optimizer.zero_grad()
                real_data = batch
                noise = torch.randint(0, self.vocab_size, (batch.size(0), self.seq_length), device=self.device)
                fake_data = self.generator(noise).detach()
                fake_data = F.softmax(fake_data, dim=-1)

                real_output =self.discriminator(real_data)
                fake_output = self.discriminator(fake_data.argmax(dim=-1))

                loss = None
                # Real labels are 1, fake labels are 0
                real_labels = torch.ones_like(real_output)
                fake_labels = torch.zeros_like(fake_output)

                if self.use_wgan:
                    # BCEWithLogitsLoss combines Sigmoid layer followed by BCELoss for numerical stability and efficiency
                    # Compute BCELoss for both real and fake outputs
                    loss = self.bce_loss(real_output, real_labels) + self.bce_loss(fake_output, fake_labels)
        
                else:
                    # Wasserstein GAN Loss: Critic maximizes E[D(x)] - E[D(G(z))]
                    w_loss = torch.mean(fake_output) - torch.mean(real_output) 
                    loss = w_loss
                
                loss.backward()
                optimizer.step()

                if not self.use_wgan:
                    # Gradient clipping for WGAN to enforce Lipschitz continuity
                    for param in self.discriminator.parameters():
                        param.grad.data.clamp_(-0.01, 0.01)

            
                # Compute classification accuracy of the discriminator for both real and fake samples
                real_acc = (torch.sigmoid(real_output) > 0.5).float().mean().item()
                fake_acc = (torch.sigmoid(fake_output) < 0.5).float().mean().item()

                # total accuracy
                acc = real_acc + fake_acc

                print("Pretrain D loss =", round(loss.item(), 3), "acc =", round(acc, 3))

        # save the pretrain discriminator
        torch.save(self.discriminator.state_dict(), f'pretrain_discriminator_{epochs}.pt')







if __name__ == '__main__':
    pass
