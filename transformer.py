import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# Create a causal mask to prevent attending to future tokens
def create_causal_mask(seq_len):
    # Returns a lower triangular matrix with 0 for future tokens and -inf for masking them out
    # mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    mask = torch.tril(torch.ones(seq_len, seq_len), diagonal=1).float() # lower triangle = True
    return mask


# def create_padding_mask(data):
#     # Returns a 0/1 matrix with 0 for padded tokens and 1 for non-padded tokens
#     padding_mask = (data == 0) # .unsqueeze(1).unsqueeze(2)
#     return padding_mask

def create_padding_mask(sequences, pad_token=0):
    """
    Create a padding mask for a batch of sequences.
    
    Args:
    sequences (torch.Tensor): Tensor of shape (batch_size, sequence_length).
    pad_token (int, optional): The token used for padding. Default is 0.
    
    Returns:
    torch.Tensor: Padding mask of shape (batch_size, sequence_length).
    """
    padding_mask = (sequences == pad_token)
    # print(padding_mask.float())
    return padding_mask.float()


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        
        # Create a matrix to store positional encodings (max_len x d_model)
        pe = torch.zeros(max_len, d_model)
        
        # Create a tensor of shape (max_len, 1) for the positions
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Compute the div_term (10000^(2i/d_model)) to scale the sine and cosine functions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices (2i) and cosine to odd indices (2i+1)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension to the positional encodings
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        
        # Register the positional encoding as a buffer, so it doesn't update during backprop
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to the input embeddings
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x

class MaskedSelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MaskedSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        assert (
            self.head_dim * num_heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        # Linear layers for queries, keys, and values
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(self.num_heads * self.head_dim, embed_size)

    def forward(self, values, keys, queries, attn_mask=None, scr_key_padding_mask=None):
        N = queries.shape[0]  # Batch size
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # Reshape the input into (batch_size, sequence_length, heads, head_dim)
        values = values.reshape(N, value_len, self.num_heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.num_heads, self.head_dim)

        # Compute energy scores (dot product of queries and keys)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # energy shape: (N, heads, query_len, key_len)

        # Apply the mask (padding mask and/or attention mask)
        if attn_mask is not None:
            # reshape attn mask : (N, heads, query_len, key_len)
            # Slice the attention mask to match the desired sequence length
            key_len = energy.size(-1)
            attn_mask = attn_mask[:, :, :key_len, :key_len] 
            # attn_mask = attn_mask.repeat(N, 1, 1).unsqueeze(1)
            print("shape of attn mask ", attn_mask.shape)
            
            # attn_mask = attn_mask.unsqueeze(1)
            energy = energy.masked_fill(attn_mask == 0, float("-1e3"))

        if scr_key_padding_mask is not None:
            print("padding mask: ", scr_key_padding_mask.shape, "|", energy.size())
            energy = energy.masked_fill(scr_key_padding_mask==0, float("-1e3"))

        # Apply softmax to obtain attention weights
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        # Attention shape: (N, heads, query_len, key_len)

        # Multiply the attention weights with the values
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.num_heads * self.head_dim
        )

        # Project back to the original embedding size
        out = self.fc_out(out)
        return out


# Create a function to generate the combined padding mask and attention mask
def generate_masks(seq_len, batch_size, padding_mask=None):
    # Create the attention mask (causal mask) to prevent attending to future positions
    attn_mask = torch.tril(torch.ones((batch_size,seq_len, seq_len))).unsqueeze(0).unsqueeze(0)

    # If padding mask is provided, combine it with the attention mask
    if padding_mask is not None:
        # Reshape padding mask to (batch_size, 1, 1, seq_len) for broadcasting
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
        combined_mask = attn_mask * padding_mask
    else:
        combined_mask = attn_mask

    return combined_mask



class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, forward_expansion, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MaskedSelfAttention(embed_size,num_heads)
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
        # Feed Forward
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, value, key, query, attn_mask, key_padding_mask):
        # Multi-head self-attention with masking
        attention = self.attention(value, key, query, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        
        # Add & Norm
        x = self.dropout(self.norm1(attention + query))
        
        # Feed forward
        forward = self.feed_forward(x)
        
        # Add & Norm
        out = self.dropout(self.norm2(forward + x))
        
        return out 
    
class TransformerEncoder(nn.Module):
    def __init__(self, embed_size, num_heads, num_layers, forward_expansion, dropout):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, num_heads, forward_expansion, dropout)
                for _ in range(num_layers)
            ]
        )
    
    def forward(self, x, attn_mask, key_padding_mask):
        for layer in self.layers:
            x = layer(x, x, x, attn_mask, key_padding_mask=key_padding_mask)  # In encoder, value, key, and query are the same 
        return x
    



class MaskedMultiheadAttention(nn.Module):
    def __init__(self, embed_size, num_heads, batch_first=False):
        super(MaskedMultiheadAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        self.batch_first = batch_first
        self._qkv_same_embed_dim = True  # Add this attribute

        assert (
            self.head_dim * num_heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        # Linear layers for queries, keys, and values
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_size, 3 * embed_size))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_size))
        self.fc_out = nn.Linear(self.num_heads * self.head_dim, embed_size)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.constant_(self.fc_out.bias, 0.)

    def forward(self, values, keys, queries, attn_mask=None, src_key_padding_mask=None):
        if self.batch_first:
            values = values.transpose(0, 1)
            keys = keys.transpose(0, 1)
            queries = queries.transpose(0, 1)

        N = queries.shape[0]  # Batch size
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # Linear projections
        qkv = torch.cat([queries, keys, values], dim=-1)
        # print("The shape of qkv is ", qkv.shape)
        # print("The shape of in_proj_bias is ", self.in_proj_bias.shape)
        qkv = torch.nn.functional.linear(qkv, self.in_proj_weight.T, self.in_proj_bias)
        queries, keys, values = qkv.chunk(3, dim=-1)

        # Reshape the input into (batch_size, sequence_length, heads, head_dim)
        values = values.reshape(N, value_len, self.num_heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.num_heads, self.head_dim)

        # Compute energy scores (dot product of queries and keys)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # energy shape: (N, heads, query_len, key_len)

        N, H, key_len, _ = energy.size()


        # Apply the mask (padding mask and/or attention mask)
        if attn_mask is not None:
            # print("shape of attn mask before resshape ", attn_mask.shape)
            # attn_mask = attn_mask[:,:, :key_len, :key_len]
            attn_mask = attn_mask.repeat(N, H, 1, 1).to(energy.device) # .unsqueeze(1)
            # print("shape of attn mask ", attn_mask.shape)
            # print("shape of energy mask ", energy.shape)
            energy = energy.masked_fill(attn_mask == 0, float("-1e3"))

        if src_key_padding_mask is not None:
            # Correctly expand the src_key_padding_mask to match the energy dimensions
            src_key_padding_mask = src_key_padding_mask.unsqueeze(1).unsqueeze(2)
            src_key_padding_mask = src_key_padding_mask.expand(-1, energy.size(1), energy.size(2), -1)
            # print("TE SHAPE OF THE PADDING MASK IS:", src_key_padding_mask.size(), "| shape of the energy =", energy.size())
            energy = energy.masked_fill(src_key_padding_mask == 0, float("-1e3"))

        # Apply softmax to obtain attention weights
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        # Attention shape: (N, heads, query_len, key_len)

        # Multiply the attention weights with the values
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.num_heads * self.head_dim
        )

        # Project back to the original embedding size
        out = self.fc_out(out)

        if self.batch_first:
            out = out.transpose(0, 1)

        return out



class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, embed_size, num_heads, dim_feedforward=2048, dropout=0.1, batch_first=False):
        super(CustomTransformerEncoderLayer, self).__init__(embed_size, num_heads, dim_feedforward, dropout, batch_first=batch_first, activation=F.relu)
        self.attn = MaskedMultiheadAttention(embed_size, num_heads, batch_first=batch_first)
        self.linear1 = nn.Linear(embed_size, dim_feedforward, )
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_size)

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.batch_first = batch_first

    def forward(self, src, src_mask=None, src_key_padding_mask=None, **kwargs):
        if self.batch_first:
            src = src.transpose(0, 1)

        # Self-attention block
        src2 = self.attn(src, src, src, attn_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feedforward block
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        if self.batch_first:
            src = src.transpose(0, 1)

        return src



if __name__ == '__main__':
    batch_size = 8
    seq_len = 50
    vocab_size = 100
    embed_size = 64 # 512
    num_heads = 4
    dim_feedforward = 512 
    dropout = 0.5
    attn_mask = create_causal_mask(seq_len) #.float()


    # Create random input tensor (batch_size, seq_len, embed_size)
    tokens = torch.randint(0, vocab_size, size=(batch_size, seq_len))
    # create an embedding tensor (batch_size, seq_len)
    embedding = nn.Embedding(vocab_size, embedding_dim=embed_size,)
    data = embedding(tokens)

    padding_mask = create_padding_mask(tokens)



    # create CustomTransformer Encoder Layer
    encoder_layer = CustomTransformerEncoderLayer(embed_size, num_heads, dim_feedforward, dropout, batch_first=False)

    # create CustomTransformer Encoder
    custom_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    output = custom_encoder(data, mask=attn_mask, src_key_padding_mask=padding_mask)

    print("Transformer Encoders output shape =", output.shape)
    

