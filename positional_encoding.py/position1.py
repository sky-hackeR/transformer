import torch
import torch.nn as nn
import math
from torch.nn.utils.rnn import pad_sequence

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)

        pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))

        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :].requires_grad_(False)
        return self.dropout(x)

# Example usage
tokenized_data = [
    [1, 2, 3, 4, 5, 6],
    [7, 8, 9, 12,6]  # Adjusted this sequence to have the same length as others
]

vocab_size = 13  # Increased to accommodate the number 12
d_model = 8
seq_len = 7
dropout = 0.1

# Convert tokenized data to a list of tensors
tokenized_tensors = [torch.tensor(seq) for seq in tokenized_data]

# Pad the sequences to the same length
padded_sequences = pad_sequence(tokenized_tensors, batch_first=True)
print("Padded sequences:\n", padded_sequences)

# Create an embedding layer
embedding_layer = nn.Embedding(vocab_size, d_model)
embedded_data = embedding_layer(padded_sequences)
print("Embedded data:\n", embedded_data)

# Create the positional encoding layer
pos_encoder = PositionalEncoding(d_model, seq_len, dropout)

# Apply the positional encoding
encoded_data = pos_encoder(embedded_data)
print("Encoded data:\n", encoded_data)
