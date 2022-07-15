import torch
import numpy as np
import gensim
from .vocab import Vocab

def gensim_w2v_to_pytorch(vocab, wv):
    emb_dim = wv[0].shape[0]
    emb_matrix = np.zeros((len(vocab), emb_dim))
    wv_keys = wv.key_to_index.keys()
    for i in range(len(vocab)):
        if str(i) in wv_keys:
            emb_matrix[i] = wv[str(i)]
        elif i == vocab.pad_index:
            pass
        else:
            emb_matrix[i] = np.random.normal(size=64)
    return emb_matrix
        

if __name__  == '__main__':
    
    wv_path = "../data/en_char_emb.wv"
    dict_path = "../data/en_dictionary.txt"
    
    vocab = Vocab(dict_path)
    wv = gensim.models.KeyedVectors.load_word2vec_format(wv_path)
    
    emb_matrix = gensim_w2v_to_pytorch(vocab, wv)    
    

    
    
    