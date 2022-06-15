
import torch
import torch.nn as nn 

class RNNLM(nn.Module):
    
    def __init__(self, voc_embd, voc_size, enc_size, hid_size, dec_size, n_layer, drop_rate, **kwargs) -> None:
        super(RNNLM, self).__init__()
        
        self.embedding_layer = voc_embd
        
        self.encoder = nn.LSTM(enc_size,
                               hid_size,
                               n_layer,
                               batch_first=True)
        
        self.decoder = nn.LSTM(hid_size,
                               dec_size,
                               n_layer,
                               batch_first=True)
        self.linear = nn.Linear(dec_size, voc_size)
        
        self.drop = nn.Dropout(drop_rate)
        
    def forward(self, x, h):
        
        
        
        