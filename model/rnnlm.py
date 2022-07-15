
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
        
        self.decoder = nn.LSTM(hid_size,
                               dec_size,
                               n_layer,
                               batch_first=True)
        self.linear = nn.Linear(hid_size, voc_size)
        
        self.drop = nn.Dropout(drop_rate)
        
        self.tied_weight()
        
        
    
    def init_hidden(self, batch_size, n_seq):
        # weight = next(self.parameters())
        # return weight.new_zeros(batch_size, n_seq, self.hid_size)
        return (torch.zeros(self.n_layer, batch_size, self.hid_size),
                torch.zeros(self.n_layer,batch_size, self.hid_size))
    
    def tied_weight(self):
        self.linear.wegiht = self.embedding_layer.encoder.weight
  
    def freeze_emb(self):
        self.linear.weight.requires_grad = False
        self.embedding_layer.encoder.weight.requires_grad = False
    
    def copy_emb(self):
        self.linear.weight = self.embedding_layer.encoder.weight.detach().clone()
        self.embedding_layer.encoder.weight.requires_grad = False
    
    def forward(self, x, h):
        
        emb = self.drop(self.embedding_layer(x))
        out, h = self.encoder(emb, h)
        out = self.drop(out)
        out, _ = self.decoder(out)
        out = self.drop(out)
        
        logit = self.linear(out)
        logit = logit.view(-1, self.voc_size)
        logit = nn.functional.log_softmax(logit, dim=-1)
        
        return logit, h 
        
        