
import torch
import torch.nn as nn 

class RNNLM(nn.Module):
    
    def __init__(self, vocab_emb, vocab_size, enc_size, hid_size, dec_size, n_layer, drop_rate) -> None:
        super(RNNLM, self).__init__()
        
        self.embedding_layer = vocab_emb
        
        self.encoder = nn.LSTM(enc_size,
                               hid_size,
                               n_layer,
                               batch_first=True)
        
        self.decoder = nn.LSTM(hid_size,
                               dec_size,
                               n_layer,
                               batch_first=True)
        self.linear = nn.Linear(dec_size, vocab_size)
        
        self.drop = nn.Dropout(drop_rate)
        
    def forward(self, x, h):
        
        
        
        