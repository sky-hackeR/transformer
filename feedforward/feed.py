import torch
import torch.nn as nn 
import math
from torch.nn.utils.rnn import pad_sequence


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model:int,dff:int , dropout:float):
        super().__init__()
        self.d_model = d_model
        self.dff = dff
        self.linear_1 = nn.Linear(d_model,dff)  #w1 and b1
        self.dropout= nn.Dropout(dropout)
        self.linear_2 = nn.Linear(dff,d_model)  #w2 and b2

    def forward(self,x):
        #(Batch,seq_len, d_model) --> (Batch, seq_len, d_ff) --> (Batch,seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    

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

d_model = 9
seq_len = 9
dropout = 0.1
dff = 32


ten_data = [torch.tensor(seq) for seq in tokenized_data]
pad = pad_sequence(ten_data, batch_first=True).float()
print("pad", pad)

feed = FeedForwardBlock(d_model,dff,dropout)
fd= feed.forward(pad)
print("fd",fd)



