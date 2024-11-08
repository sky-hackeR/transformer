import torch
import torch.nn as nn
import math

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model:int, h:int, dropout:float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropout = dropout
        assert d_model % h == 0, "d_model must be divisible by h"

        self.d_k = d_model // h 

        self.w_q = nn.Linear(d_model, d_model) # wq
        self.w_k = nn.Linear(d_model, d_model) # wk
        self.w_v = nn.Linear(d_model, d_model) # wv

        self.w_o = nn.Linear(d_model, d_model) # wo
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return attention_scores @ value, attention_scores

    def forward(self, q, k, v, mask=None):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        return self.w_o(x)


# Hyperparameters
d_model = 16  # dimensionality of the input and output
h = 4         # number of attention heads
dropout = 0.1 # dropout rate
batch_size = 2
seq_len = 10

# Initialize the multi-head attention block
mha = MultiHeadAttentionBlock(d_model, h, dropout)

# Create random tensors to represent the query, key, and value inputs
q = torch.randn(batch_size, seq_len, d_model)
k = torch.randn(batch_size, seq_len, d_model)
v = torch.randn(batch_size, seq_len, d_model)

# Create a random mask tensor (optional)
mask = torch.randint(0, 2, (batch_size, 1, 1, seq_len))

# Pass the inputs through the multi-head attention block
output = mha(q, k, v, mask)

print("Query (q):", q)
print("Key (k):", k)
print("Value (v):", v)
print("Mask:", mask)
print("Output:", output)
