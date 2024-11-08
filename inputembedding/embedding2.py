import torch 
import torch.nn as nn
import math 
from torch.nn.utils.rnn import pad_sequence


class InputEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    

d_model = 6

text_data = [
    "This is an example sentence.",
    "We are tokenizing and embedding this text.",
    "Each word will be represented by a token."
]

tokens = []
word_to_index = {}
index = 0
tokenized_data = []
for sentence in text_data:
    sentence_tokens = []
    for word in sentence.split():
        if word not in word_to_index:
            word_to_index[word] = index
            index += 1
        sentence_tokens.append(word_to_index[word])
    tokenized_data.append(sentence_tokens)

# Vocabulary size
vocab_size = len(word_to_index)

# Convert tokenized data to a list of tensors
tokenized_tensors = [torch.tensor(seq) for seq in tokenized_data]
print("tokenized_tensors", tokenized_tensors)

# Pad the sequences to the same length
padded_sequences = pad_sequence(tokenized_tensors, batch_first=True)
print("padded sequences", padded_sequences)

inputembedding = InputEmbedding(vocab_size, d_model)

embd = inputembedding(padded_sequences)

print("embedded", embd)
