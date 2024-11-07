import argparse
import csv
import dis

import lightning as L
import numpy as np
import torch

from data_iter import DisDataModule, GenDataModule
from file_reader import load_and_print_dataset
from tengan import Discriminator, Generator, SequenceGenerator, TenGAN
import tengan_config
from tokenizer import Tokenizer


def fetch_malware_samples(sequences, labels):
    return [sequences[idx] for idx, label  in enumerate(labels) if label=="malware"]

def clean_sequences(gen_samples, tokenizer, token2int=False):
    cleaned_sequences = []
    for seq in gen_samples:
        # Remove first token
        seq = seq[1:]
        try:
            idx_end_token = seq.index(tokenizer.token_to_int[tokenizer.end])
        except ValueError:
            idx_end_token = None  # or any other value indicating the item is not found

        if idx_end_token is not None:
            seq = seq[:idx_end_token]
        
        if token2int:
            seq = [tokenizer.int_to_token[token] for token in seq if tokenizer.int_to_token[token] not in [tokenizer.end, tokenizer.start, tokenizer.pad]]
        cleaned_sequences.append(np.array(seq, dtype=np.int32).tolist())
    
    return cleaned_sequences
def read_generated_samples(filename, clean=False, tokenizer=None, token2int=False):
    # Read from CSV file
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        data = [row for row in reader]

    gen_data = np.array(data, dtype=np.int32).tolist()
    if clean:
        assert tokenizer is not None, "Tokenizer cannot be None to get clean sequences."
        gen_data = clean_sequences(gen_data, tokenizer, token2int=token2int)
    
    return gen_data

def remove_item_and_repetitions(lst, item):
    return [x for x in lst if x != item]
    


def main():
    
    args = tengan_config.get_args()

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

    else:
        # load weights to the generator 
        # gen_checkpoint_path = "lightning_logs/version_2/checkpoints/epoch=9-step=660.ckpt"
        # gen_checkpoint_path = "lightning_logs/version_16/checkpoints/epoch=94-step=3135.ckpt"

        gen_checkpoint_path = "lightning_logs/version_20/checkpoints/epoch=49-step=1650.ckpt"

        generator = Generator.load_from_checkpoint(checkpoint_path=gen_checkpoint_path) 

    
    sequence_sampler = SequenceGenerator(generator, tokenizer, args.seq_length, args.batch_size)

    # generate samples for pre-training the discriminator
    n = len(malware_train_sequences)
    filename = "data/generated_data.csv"
    # sequence_sampler.sample_multi(n=n,filename=filename)

    
    if args.pre_training_disc:
        # generate multiple samples and save them in a file.
        # n = len(malware_train_sequences)
        # sequence_sampler.sample_multi(n=n,filename=filename)
        # step 2: Pretrain discriminator [if pretrain_disc is True]   
        disc_trainer = L.Trainer(max_epochs=args.pretrain_epochs, enable_progress_bar=True, precision=16, log_every_n_steps=n_steps, accelerator=accelerator)
        
        # grab negative samples
        gen_malware_sequences = read_generated_samples(filename, clean=True, tokenizer=tokenizer, token2int=True)

        disc_data_module = DisDataModule(malware_train_sequences, gen_malware_sequences, max_seq_len=args.seq_length,batch_size=args.batch_size, tokenizer=tokenizer)
        # print(disc_dataloader.train_dataloader())
    
        disc_data_module.setup()
        disc_trainer.fit(discriminator, train_dataloaders=disc_data_module.train_dataloader())

        # save pretrained generator
        torch.save(generator.state_dict(), f"saved_models/discriminator_{args.pretrain_epochs}.pth")
    
    else:
        # discriminator = Discriminator(vocab_size=len(tokenizer.tokenlist), embedding_dim=args.embedding_dim, num_heads=args.num_heads, num_layers=args.num_encoders, dim_feedforward=args.dim_feedforward, seq_length=args.seq_length, dropout=args.dropout)
        # dis_checkpoint_path = "lightning_logs/version_17/checkpoints/epoch=14-step=885.ckpt"
        dis_checkpoint_path = "lightning_logs/version_21/checkpoints/epoch=49-step=1700.ckpt"
        # hparams_file = "lightning_logs/version_20/checkpoints/hparams.yaml"
        discriminator = Discriminator.load_from_checkpoint(checkpoint_path=dis_checkpoint_path)
        # model_path = "saved_models/discriminator_20.pth"
        # discriminator.load_state_dict(torch.load(model_path))

        # discriminator = torch.load(dis_checkpoint_path)

    if args.adversarial_training:
        # step 3: train the tengan model 
        real_data_module = GenDataModule(malware_train_sequences,max_seq_len=args.seq_length, train_size=len(malware_train_sequences), batch_size=args.batch_size,tokenizer=tokenizer)
        real_data_module.setup()
        # gan_model = TenGAN(generator, discriminator, sequence_sampler, args)
        gan_model = TenGAN(generator, discriminator, tokenizer, args)
        # trainers for gan model
        
        gan_trainer = L.Trainer(enable_progress_bar=True, precision=16, log_every_n_steps=n_steps, accelerator="mps", max_epochs=args.epochs) # accumulate_grad_batches=4

        # gan_trainer.fit(gan_model, datamodule=real_data_module)
        gan_trainer.fit(gan_model, train_dataloaders=real_data_module.train_dataloader()) 
        # gan_model = gan_model
    # for batch_idx, batch in enumerate(tqdm(real_data_module.train_dataloader())):
    #     gan_model.training_step(batch, batch_idx)



if __name__ == "__main__":
    main()  

    # 1) pretrain generator:python3 run_tengan.py --pretrain_epochs=50 --pre_training_gen=True
    # 2) pretrain discriminator, generate samples, then :python3 run_tengan.py --pretrain_epochs=20 --pre_training_disc=True
    # 3) adversarial training:python3 run_tengan.py --pretrain_epochs=20 --pre_training_gen=False --pre_training_disc=False --adversarial_training=True