
import torch
import torch.nn as nn 

class RNNLM(nn.Module):
    
    def __init__(self, voc_embd, voc_size, enc_size, hid_size, dec_size, n_layer, drop_rate, tie_weight, **kwargs) -> None:
        super(RNNLM, self).__init__()
        
        self.n_layer = n_layer
        self.hid_size = hid_size
        self.voc_size = voc_size
        
        self.embedding_layer = voc_embd
        
        self.encoder = nn.LSTM(enc_size,
                               hid_size,
                               n_layer,
                               batch_first=True)
        
        # self.decoder = nn.LSTM(hid_size,
        #                        dec_size,
        #                        n_layer,
        #                        batch_first=True)
        self.linear = nn.Linear(hid_size, voc_size)
        
        self.drop = nn.Dropout(drop_rate)
        
        
    
    def init_hidden(self, n_seq):
        weight = next(self.parameters())
        return weight.new_zeros(self.n_layer, n_seq, self.hid_size)
    
    
    def forward(self, x, h):
        
        emb = self.drop(self.embedding_layer(x))
        
        out, h = self.encoder(emb, h)
        out = self.drop(out)
        
        logit = self.linear(out)
        logit = logit.view(-1, self.voc_size)
        
        
        