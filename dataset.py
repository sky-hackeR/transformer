import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.ds = ds 
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64)
        self.end_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
       src_target_pair = self.ds[idx]
       src_text = src_target_pair['translation'][self.src_lang]
       tgt_text = src_target_pair['translation'][self.tgt_lang]
       
       enc_input_tokens = self.tokenizer_src.encode(src_text).ids            #gives input ids pointing to each word in the original sentence 
       dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

       enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
       dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1        #decoding part start of sentence tokens

       if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:           #sequence length must be sufficient 
           raise ValueError("Sentence is too long")
       
       #Tensor for encoder input and decoder input and output or target = label
       
       #Add SOS and EOS to the Source text
       
       encoder_input = torch.cat(
           [
           self.sos_token,
           torch.tensor(enc_input_tokens, dtype=torch.int64),
           self.end_token,
           torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
           ]
        )
       
       #Add SOS to decoder input
       decoder_input = torch.cat([
           self.sos_token,
           torch.tensor(dec_input_tokens, dtype=torch.int64),
           torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
       ])

       #label side we have end of sentence tokens EOS (what we expect as output from the decoder)

       label = torch.cat([
           torch.tensor(dec_input_tokens, dtype=torch.int64),
           self.end_token,
           torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
       ])

       assert encoder_input.size(0)  == self.seq_len
       assert decoder_input.size(0) == self.seq_len
       assert label.size(0) == self.seq_len

       return {
           'encoder_input': encoder_input,   # (Seq_len),
           "decoder_input": decoder_input,   # (Seq_len)
           "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), #(1,1, seq_len)
           "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),  #(1, seq_len) & (1, seq_len, seq_len) the decoder mask word can only look at the (word before) previous word and known padding words (we don't want padding tokens to participate, just real words)
           "label": label, #(seq_len)
           "src_text": src_text,
           "tgt_text": tgt_text,
       }

def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0 # everything below the diagonal to become true = 1 and above the diagonal to become = 0
