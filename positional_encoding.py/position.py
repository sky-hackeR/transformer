import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))

        # Apply sin to even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cos to odd positions
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)
        print("peing",pe)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :].requires_grad_(False)  # Make the parameter not learnable
        return self.dropout(x)

# Example usage
vocab_size = 11
d_model = 16
seq_len = 10
dropout = 0.1

# Example tokenized data (batch of 2 sequences of length 10)
tokenized_data = torch.tensor([
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
])

# Create an embedding layer
embedding_layer = nn.Embedding(vocab_size, d_model)
embedded_data = embedding_layer(tokenized_data)

print("Embedded data:")
print(embedded_data)

# Create the positional encoding layer
pos_encoder = PositionalEncoding(d_model, seq_len, dropout)

# Apply the positional encoding
encoded_data = pos_encoder(embedded_data)

print("Encoded data:")
print(encoded_data)
