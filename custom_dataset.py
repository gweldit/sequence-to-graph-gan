

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class CustomSequenceDataset(Dataset):
    def __init__(self, data, labels, length=None):
        self.data = data
        self.labels = labels
        self.length = length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.length is not None:
            sample = self.data[idx][:self.length]
        
        return sample, label
    

def collate_fn(batch):
    # Sort batch by sequence length (descending)
    batch.sort(key=lambda x: len(x[0]), reverse=True)

    # Separate sequences and labels
    sequences, labels = zip(*batch)
    
    # Convert sequences to tensors and pad them
    sequences = [torch.tensor(seq) for seq in sequences]
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    
    # Convert labels to a tensor
    labels = torch.tensor(labels)

    # Compute lengths of each sequence (useful for packing)
    lengths = torch.tensor([len(seq) for seq in sequences])

    return padded_sequences, labels, lengths

