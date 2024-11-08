import torch
import torch.nn as nn 
import math 
from torch.nn.utils.rnn import pad_sequence

class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, seq_len:int, dropout:float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout= nn.Dropout(dropout)

        pe = torch.zeros(seq_len,d_model)

        pos = torch.arange(0, seq_len).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))

        pe[:,0::2]= torch.sin(pos * div_term)
        pe[:,1::2] = torch.cos(pos * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe",pe)

    def forward(self,x):
        x = x + self.pe[:, :x.shape[1], :].requires_grad_(False)
        return self.dropout(x)
    


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

d_model = 6
seq_len = 9
dropout = 0.1

data_ten = [torch.tensor(seq) for seq in tokenized_data]
pad = pad_sequence(data_ten,batch_first=True)
print("pad",pad)

embedding = nn.Embedding(vocab_size, d_model)
embd = embedding(pad)
print("Embedding",embd)

pos = PositionalEncoding(d_model,seq_len,dropout)
new = pos(embd)

print("new=>",new)
