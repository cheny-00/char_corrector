
import torch
import torch.nn as nn
import torch.nn.functional as F

# from https://github.com/kamigaito/rnnlm-pytorch/blob/master/models.py
class CNNCharEmb(nn.Module):
    """
    CNN for embedding characters
    """
    def  __init__(self, prm):
        super(CNNCharEmb, self).__init__()
        self.prm = prm
        self.encoder = nn.Embedding(prm["voc_size"], prm["emb_size"])
        self.drop = nn.Dropout(prm["drop_rate"])
        self.conv_layers = nn.ModuleList([nn.Conv1d(prm["emb_size"], prm["hid_size"], kernel_size=ksz, padding=(prm["char_kmax"]-prm["char_kmin"])) for ksz in range(prm["char_kmin"], prm["char_kmax"]+1)])
        self.fullcon_layer = nn.Linear(prm["hid_size"]*(prm["char_kmax"] - prm["char_kmin"] + 1), prm["hid_size"])

    def forward(self, inp_data):
        """
        Calculate embeddings of characters
        
        Inputs
        ----------
        input: [seq_len*nbatch, token_len]
        
        Returns
        ----------
        emb: [seq_len*nbatch, char_hid]
        """
        # char_emb: [seq_len*nbatch, token_len, char_emb]
        char_emb = self.drop(self.encoder(inp_data))
        return char_emb
        print("char emb", char_emb.shape)
        list_pooled = []
        """ calculate convoluted hidden states of every kernel """
        for ksz in range(self.prm["char_kmin"], self.prm["char_kmax"]+1):
            # print(char_emb.shape)
            conved = self.conv_layers[ksz - 1](char_emb.permute(0,2,1))
            # print(conved.shape)
            list_pooled.append(F.max_pool1d(conved,kernel_size=conved.shape[1]).squeeze(2))
            
        # pooled: [seq_len*nbatch, char_hid]
        pooled = torch.cat(list_pooled, dim=1)
        print("pooled emb", pooled.shape)
        return pooled
        # word_emb: [seq_len*nbatch, char_hid]
        # word_emb = torch.tanh(self.fullcon_layer(pooled))
        
        # return word_emb