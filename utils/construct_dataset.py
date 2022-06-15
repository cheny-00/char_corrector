import os

import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset

import utils
import data_preprocess
from utils.vocab import Vocab

"""
TODO:
1. read data
2. random sample
3. normalize
4. create vocab 
5. save samples
6. 
"""
# TODO support generator

class BaseDataset(Dataset):
    
    def __init__(self, corpus, vocab, seq_len, encoding="utf-8", corpus_lines=None, on_memory=True):
        
        self.vocab = vocab
        self.seq_len = seq_len

        self.on_memory = on_memory
        self.corpus_lines = corpus_lines
        self.corpus = corpus
        self.encoding = encoding
        

    

class LMDataset(BaseDataset):
    
    def __init__(self, corpus, vocab, seq_len, encoding="utf-8", corpus_lines=None, on_memory=True, n_corpus=None):
        super().__init__(corpus, vocab, seq_len, encoding, corpus_lines, on_memory)
        
        if n_corpus is not None and n_corpus != corpus_lines:
            self.corpus = utils.reservoir_sampling(self.corpus, n_corpus)
            self.corpus_lines = n_corpus
        self.corpus = data_preprocess.normalize_sents(self.corpus)
        self.corpus = self.word_2_indices(self.corpus, vocab)
        # print(self.corpus, 'zz')
        
    @staticmethod
    def word_2_indices(corpus, vocab):

        indices_corpus = list()
        for line in corpus:
            indices_line = list()
            line = line.split(' ')
            for char in line:
                char_index = vocab.stoi.get(char, vocab.unk_index)
                if char_index == vocab.unk_index:
                    print(char)
                indices_line.append(char_index)
            indices_corpus.append(indices_line)
        return indices_corpus
                
    
    def __len__(self):
        return self.corpus_lines
    
    @staticmethod
    def patch_data(corpus, idx):
        
        features = dict()
        features['x'] = corpus[idx]
        return features
    
    def __getitem__(self, index):
        sample = self.patch_data(self.corpus, index)
        return sample
        
    
if __name__ == '__main__':
    
    import datasets
    corpus_path = '/mnt/d/datasets/en/bookcorpus/saved/train'
    vocab_path = '../data/en_dictionary.txt'

    corpus_data = datasets.load_from_disk(corpus_path)
    vocab = Vocab(vocab_path)
    seq_len = 128
    
    dataset = LMDataset(
        corpus=corpus_data,
        vocab=vocab,
        seq_len=seq_len,
        corpus_lines=None,
        
    )
    
    
    
    