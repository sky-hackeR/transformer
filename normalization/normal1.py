import torch
import torch.nn as nn
import math
from torch.nn.utils.rnn import pad_sequence

class LayerNormalization(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        self.alpha = nn.Parameter(torch.ones(x.size(-1)))
        self.bias = nn.Parameter(torch.zeros(x.size(-1)))

        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        return self.alpha * (x - mean) / torch.sqrt(var + self.eps) + self.bias

text_data = [
    "This is an example sentence kolawole olanrewaju sunday adeseun",
    "We are tokenizing and embedding This text.",
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

d_model = 8
seq_len = 9
dropout = 0.1

ten_data = [torch.tensor(seq) for seq in tokenized_data]
pad = pad_sequence(ten_data, batch_first=True).float()
print("pad", pad)

lay = LayerNormalization()
la1 = lay.forward(pad)
print("Layer normalized output:", la1)
