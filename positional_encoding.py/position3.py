import torch
import torch.nn as nn
import math
from torch.nn.utils.rnn import pad_sequence


class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int,seq_len:int,dropout:float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout= nn.Dropout(dropout)

        pe = torch.zeros(seq_len,d_model)

        pos = torch.arange(0, seq_len, dtype=float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))

        pe[:,0::2] = torch.sin(pos * div_term)
        pe[:,1::2] = torch.cos(pos * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe",pe)

    def forward(self,x):
        x = x + self.pe[:, :x.shape[1], :].requires_grad_(False)
        return self.dropout(x)
    

vocab_size = 12
seq_len = 16
d_model= 18
dropout = 0.1

tokenized = [[1,2,3,4,4],
             [5,6,7,8],
             [9,8,3]

]

tensor_data = [torch.tensor(seq) for seq in tokenized]

pad = pad_sequence(tensor_data,batch_first=True)
print("pad",pad)

emdd = nn.Embedding(vocab_size,d_model)

tk = emdd(pad)

print("padding==>", tk)



post = PositionalEncoding(d_model,seq_len,dropout)

ps = post(tk)

print("postional endoing ==>",ps )





