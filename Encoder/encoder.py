import torch 
import torch.nn as nn
import math 



class EncoderBlock(nn.Module):

    def __init__(self,self_attention_block:nn.MultiHeadAttentionBlock, feed_forward_block:nn.FeedForwardBlock, dropout:float):
        super().__init__()