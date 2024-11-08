import torch 
import torch.nn as nn
import math
from torch.nn.utils.rnn import pad_sequence

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size:int , d_model:int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(d_model,vocab_size)


    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)
    


vocab_size = 50
d_model = 5

tokenized_data = [[1,2,3,4,5],
                  [6,7,8,9,1],
                  [10,11,13],
                  [13,34,27]]


tensor_data = [torch.tensor(seq) for seq in tokenized_data]

print("tokenized_data =>",tokenized_data)

padding_tk = pad_sequence(tensor_data,batch_first=True)
print("padding_tokens",padding_tk)


input_embedding = InputEmbedding(d_model,vocab_size)

embd = input_embedding(padding_tk)

print("embedded", embd)