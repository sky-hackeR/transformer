import torch 
import torch.nn as nn
import math
from torch.nn.utils.rnn import pad_sequence


class Transformer(nn.Module):
    def __init__(self,vocab_size:int,d_model:int,seq_len:int,dropout:float):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model= d_model 
        self.seq_len = seq_len 
        self.embedding = nn.Embedding(vocab_size,d_model)
        self.dropout = nn.Dropout(dropout)

    def Input_embeding(self,x):
        return self.embedding(x) * math.sqrt(d_model)
    
    def position_encoding(self,x):
        pe = torch.zeros(seq_len,d_model)

        pos = torch.arange(0,seq_len).unsqueeze(1)

        div_term =  torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))

        pe[:,0::2] = torch.sin(pos * div_term)
        pe[:,1::2] = torch.cos(pos * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe",pe)

        x = x + self.pe[:, :x.shape[1], :].requires_grad_(False)
        return self.dropout(x)
    

    def LayerNormalization(self,x):
        eps = 1e-6
        self.alpha = nn.Parameter(torch.ones(x.size(-1)))
        self.bias = nn.Parameter(torch.zeros(x.size(-1)))

        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        return self.alpha * (x - mean) / torch.sqrt(var + eps) + self.bias

    def feedforwardblock(self,x):
        dff = 32
        self.linear_1 = nn.Linear(d_model,dff)
        self.dropout= nn.Dropout(dropout)
        self.linear_2 = nn.Linear(dff,d_model)


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

d_model = 8
seq_len = 9
dropout = 0.1


trans = Transformer(vocab_size,d_model,seq_len,dropout)

ten_data = [torch.tensor(seq) for seq in tokenized_data]

pad = pad_sequence(ten_data)
print("pad",pad)
embd= trans.Input_embeding(pad)

print("padding",embd)


tk = trans.position_encoding(embd)
print("tokens",tk)

layer = trans.LayerNormalization(tk)
print("layers",layer)

feed = trans.feedforwardblock(layer)
print("feed",feed)



