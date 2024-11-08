import torch 
import torch.nn as nn
import math 
from torch.nn.utils.rnn import pad_sequence

 
class InputEmbedding(nn.Module):
    def __init__(self, vocab_size:int, d_model:int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    

vocab_size = 16
d_model = 6

tokenized_data = [
    [4, 12, 6, 8, 9,8],    
    [2, 5, 7, 10, 3, 11],
    [15, 14, 12, 1] 
]

# Convert tokenized data to a list of tensors
tokenized_tensors = [torch.tensor(seq) for seq in tokenized_data]
print("tokenized_tensors", tokenized_tensors)

# Pad the sequences to the same length
padded_sequences = pad_sequence(tokenized_tensors, batch_first=True)
print("padded sequences", padded_sequences)

inputembedding = InputEmbedding(vocab_size, d_model)

embd = inputembedding(padded_sequences)

print("embedded", embd)
