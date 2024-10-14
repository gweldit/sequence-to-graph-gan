import argparse
import csv

import lightning as L
import numpy as np
import torch

from data_iter import DisDataModule, GenDataModule
from file_reader import load_and_print_dataset
from tengan import Discriminator, Generator, SequenceGenerator, TenGAN
from tokenizer import Tokenizer


def fetch_malware_samples(sequences, labels):
    return [sequences[idx] for idx, label  in enumerate(labels) if label=="malware"]


def read_samples(filename):
    # Read from CSV file
    with open('output.csv', 'r') as file:
        reader = csv.reader(file)
        data = [row for row in reader]

    gen_data = np.array(data, dtype=np.int32).tolist()
    return gen_data

def remove_item_and_repetitions(lst, item):
    return [x for x in lst if x != item]
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    # parser.add_argument('--vocab_size', type=int, default=200)
    parser.add_argument('--seq_length', type=int, default=290) # 300, 400
    parser.add_argument('--embedding_dim', type=int, default=64) # 512
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_encoders', type=int, default=3) # 4
    parser.add_argument('--dim_feedforward', type=int, default=64) # 12
    parser.add_argument('--dropout', type=float, default=0.1)
    # parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--pretrain_epochs', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--use_wgan', type=bool, default=False)
    parser.add_argument('--n_critic', type=int, default=5)
    parser.add_argument('--g_steps', type=int, default=3) # higher value is better at estimating and reducing variance in gradients

    parser.add_argument('--lr', type=float, default=0.0002)

    parser.add_argument('--train_file_path', type=str, help='The path to the file containing the sequences of training dataset',default="data/train_dataset.json") 


    parser.add_argument('--test_file_path', type=str, help='The path to the file containing the sequences of testing set',default="data/test_dataset.json") 
    parser.add_argument('--pre_training_disc', type=bool, default=False)
    parser.add_argument('--pre_training_gen',type=bool, default=False)
    parser.add_argument('--adversarial_training',type=bool, default=False)

    args = parser.parse_args()
    




    # stpe 1: Load sequences dataset and fetch only positive (malware samples)

    train_sequences, train_labels = load_and_print_dataset(args.train_file_path, print_data=False)

    test_sequences, test_labels = load_and_print_dataset(args.test_file_path, print_data=False)

    # return only positive samples
    malware_train_sequences = fetch_malware_samples(train_sequences, train_labels)


    malware_test_sequences = fetch_malware_samples(test_sequences, test_labels)


    # step 2: Tokenize the dataset 
        # a) combine all sequences for the tokenizer
        # b) build vocabulary
        # c) tokenize

    sequences = train_sequences + test_sequences

    # max_seq_len = max([len(s) for s in  malware_train_sequences])
    # args.seq_length = max(args.seq_length, max_seq_len)

    tokenizer = Tokenizer() # may pass this to all dataloaders for uniform encoding

    tokenizer.build_vocab(sequences)

    # print(tokenizer.tokenlist, len(tokenizer.tokenlist))

    # step 3: Pretrain generator [if pretrain_gen is True]
    # model = TenGAN(args)
    # args.vocab_size = min(args.vocab_size, len(tokenizer.tokenlist))

    print(args)


    generator = Generator(vocab_size=len(tokenizer.tokenlist), embedding_dim=args.embedding_dim, num_heads=args.num_heads, num_layers=args.num_encoders, dim_feedforward=args.dim_feedforward, seq_length=args.seq_length, dropout=args.dropout)

    discriminator = Discriminator(vocab_size=len(tokenizer.tokenlist), embedding_dim=args.embedding_dim, num_heads=args.num_heads, num_layers=args.num_encoders, dim_feedforward=args.dim_feedforward, seq_length=args.seq_length, dropout=args.dropout)
    
    # seed_everything(42, workers=True)
    # sets seeds for numpy, torch and python.random.
    # trainer = Trainer(deterministic=True, min_epochs=3, max_epochs=args.epochs)

    accelerator = "mps"
    n_steps = 33


    trainer = L.Trainer(max_epochs=args.pretrain_epochs, enable_progress_bar=True, precision=16, log_every_n_steps=n_steps, accelerator=accelerator)

    if args.pre_training_gen:
        # step 1: pre-training generator if pre_training_gen is True
        gen_data_module = GenDataModule(malware_train_sequences,max_seq_len=args.seq_length, train_size=len(malware_train_sequences), batch_size=args.batch_size, tokenizer=tokenizer)
        # print(gen_dataloader.train_dataloader())
    
        gen_data_module.setup()
        trainer.fit(generator, train_dataloaders=gen_data_module.train_dataloader(), val_dataloaders=gen_data_module.val_dataloader())

        # save pretrained generator
        torch.save(generator.state_dict(), f"saved_models/generator_{args.pretrain_epochs}.pth")

    # else:
    #     # load weights to the generator 
    #     # gen_checkpoint_path = "lightning_logs/version_2/checkpoints/epoch=9-step=660.ckpt"
    #     gen_checkpoint_path = "lightning_logs/version_8/checkpoints/epoch=9-step=330.ckpt"
    #     generator = Generator.load_from_checkpoint(checkpoint_path=gen_checkpoint_path)

    sequence_sampler = SequenceGenerator(generator, tokenizer, args.seq_length, args.batch_size)

    
    if args.pre_training_disc:
        # generate multiple samples and save them in a file.
        # n = len(malware_train_sequences)
        sequence_sampler.sample_multi(n=args.batch_size * 5,filename="data/generated_data.csv")
        # step 2: Pretrain discriminator [if pretrain_disc is True]   
        disc_trainer = L.Trainer( max_epochs=args.pretrain_epochs, enable_progress_bar=True, precision=16, log_every_n_steps=n_steps, accelerator=accelerator)

        disc_data_module = DisDataModule(malware_train_sequences, malware_train_sequences, max_seq_len=args.seq_length,batch_size=args.batch_size,  tokenizer=tokenizer)
        # print(disc_dataloader.train_dataloader())
    
        disc_data_module.setup()
        disc_trainer.fit(discriminator, train_dataloaders=disc_data_module.train_dataloader())

        # save pretrained generator
        torch.save(generator.state_dict(), f"saved_models/discriminator_{args.pretrain_epochs}.pth")
    # else:
    #     dis_checkpoint_path = "lightning_logs/version_4/checkpoints/epoch=0-step=118.ckpt"
    #     hparams_file = "lightning_logs/version_4/checkpoints/hparams.yaml"
    #     discriminator= Discriminator.load_from_checkpoint(checkpoint_path=dis_checkpoint_path, hparams_file=hparams_file)

    if args.adversarial_training:
        # step 3: train the tengan model 
        real_data_module = GenDataModule(malware_train_sequences,max_seq_len=args.seq_length, train_size=len(malware_train_sequences), batch_size=args.batch_size,tokenizer=tokenizer)
        real_data_module.setup()
        gan_model = TenGAN(generator, discriminator, sequence_sampler, args)
        # trainers for gan model
        
        gan_trainer = L.Trainer(enable_progress_bar=True, precision=16, log_every_n_steps=n_steps, accelerator="mps", max_epochs=args.epochs) # accumulate_grad_batches=4

        # gan_trainer.fit(gan_model, datamodule=real_data_module)
        gan_trainer.fit(gan_model, train_dataloaders=real_data_module.train_dataloader()) 
        # gan_model = gan_model
    # for batch_idx, batch in enumerate(tqdm(real_data_module.train_dataloader())):
    #     gan_model.training_step(batch, batch_idx)



if __name__ == "__main__":
    main()  