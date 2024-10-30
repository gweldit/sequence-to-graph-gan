import torch
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
# from tokenizer import Tokenizer
import numpy as np
import pandas as pd


class GenDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence = self.data[idx]
        tensor = self.tokenizer.encode(sequence)
        return tensor

class GenDataModule(LightningDataModule):

    def __init__(self, data, max_seq_len, train_size=400, batch_size=64,  tokenizer=None):
        super().__init__()
        # self.tokenizer = Tokenizer()
        self.tokenizer = tokenizer
        self.train_size = train_size
        self.val_size = int (0.1 * (batch_size))
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.data = data
    
    def custom_collate_and_pad(self, batch):
        tensors = [torch.tensor(sequence).squeeze(0) for sequence in batch]
        # for tensor in tensors:
        #     print(tensor.size())  # Print the size of each tensor
        tensors = torch.nn.utils.rnn.pad_sequence(tensors, padding_value=0, batch_first=True) # pad with zero
        tensors = tensors[:,:self.max_seq_len]
        return tensors
    
    # def custom_collate_and_pad(self, batch):
    #     max_len = max(len(sequence) for sequence in batch)
    #     tensors = [torch.tensor(sequence + [0] * (max_len - len(sequence))) for sequence in batch]
    #     tensors = torch.stack(tensors, dim=0)  # Stack tensors to create a batch
    #     return tensors

    
    def setup(self, stage=None):
        # self.tokenizer.build_vocab(self.data)
        
        idxs = np.array(range(len(self.data)))
        np.random.shuffle(idxs)
        val_idxs, train_idxs = idxs[:self.val_size], idxs[self.val_size:self.val_size + self.train_size]
        
        self.train_data = [self.data[i] for i in train_idxs]
        self.val_data = [self.data[i] for i in val_idxs]
        
    def train_dataloader(self):
        dataset = GenDataset(self.train_data, self.tokenizer)
        return DataLoader(dataset, batch_size=self.batch_size, pin_memory=True, collate_fn=self.custom_collate_and_pad, num_workers=4)
    
    def val_dataloader(self):
        dataset = GenDataset(self.val_data, self.tokenizer)
        return DataLoader(dataset, batch_size=self.batch_size, pin_memory=True, collate_fn=self.custom_collate_and_pad, shuffle=False, num_workers=4)


class DisDataset(Dataset):
    
    def __init__(self, pairs, tokenizer):
        self.data, self.labels = pairs['sequences'], pairs['labels']
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence = self.data[idx]
        tensor = self.tokenizer.encode(sequence)
        label = self.labels[idx]
        return tensor, label


class DisDataModule(LightningDataModule):

    def __init__(self, positive_data, negative_data, max_seq_len, batch_size=64, tokenizer=None):
        super().__init__()
        # self.tokenizer = Tokenizer()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.positive_data = positive_data
        self.negative_data = negative_data
    
    def custom_collate_and_pad(self, batch):
        sequences, labels = zip(*batch)
        tensors = [torch.LongTensor(seq).squeeze(0) for seq in sequences]
        tensors = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        tensors = tensors[:,:self.max_seq_len]
        labels = torch.LongTensor(labels)
        return tensors, labels
    
    def setup(self):
        self.data = self.positive_data + self.negative_data
        self.labels = [1] * len(self.positive_data) + [0] * len(self.negative_data) # real samples label = 1, fake samples label = 0.
        self.pairs = list(zip(self.data, self.labels))
        
        # self.tokenizer.build_vocab(self.data)
        
        np.random.shuffle(self.pairs)
        self.pairs = pd.DataFrame(self.pairs, columns=['sequences', 'labels'])
        
        self.train_data = self.pairs[:int(len(self.pairs) * 0.9)]
        self.train_data.reset_index(drop=True, inplace=True)
        self.val_data = self.pairs[int(len(self.pairs) * 0.9):]
        self.val_data.reset_index(drop=True, inplace=True)
        
    def train_dataloader(self):
        dataset = DisDataset(self.train_data, self.tokenizer)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, collate_fn=self.custom_collate_and_pad, num_workers=4)
    
    def val_dataloader(self):
        dataset = DisDataset(self.val_data, self.tokenizer)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, collate_fn=self.custom_collate_and_pad, num_workers=4)
