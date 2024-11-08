import torch.nn as nn
import torch 
import math

class EncoderBlock(nn.Module):
    def __init__(self,features:int, self_attention:MultiheadBlock, self_feedforward:FeedForward,dropout:float ):
        super().__init__()
        self.self_attention = self_attention
        self.self_feedforward = self_feedforward
        self.residual_connection = nn.ModuleList([ResidualConnection(features,dropout) for _ in range(2)])

    def forward(self,x, src_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention(x,x,x,src_mask))
        x = self.residual_connection[1](x, self.self_feedforward)
        return  x
    
class DecoderBlock(nn.Module):
    def __init__(self, features:int, self_attention:MultiheadBlock, cross_attention:MultiheadBlock,self_feedforward:FeedForward,dropout:float):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.self_feedforward = self_feedforward
        self.residual_connection = nn.ModuleList([ResidualConnection(features,dropout) for _ in range(3)])

    def forward(self,x, encoder_output,src_mask, tgt_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention(x,x,x, tgt_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection[2](x, self.self_feedforward)

        return x
 



