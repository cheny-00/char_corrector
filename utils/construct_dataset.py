import os

import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset

import utils
import data_preprocess

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
    
    def __init__(self, corpus_path, vocab, seq_len, encoding="utf-8", corpus_lines=None, on_memory=True):
        
        self.vocab = vocab
        self.seq_len = seq_len
        
        self.on_memory = on_memory
        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path
        self.encoding = encoding
        
        with open(corpus_path, "r", encoding=encoding) as f:
            if self.corpus_lines is None and not on_memory:
                for _ in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
                    self.corpus_lines += 1

            if on_memory:
                self.lines = [line
                              for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]
                self.corpus_lines = len(self.lines)

        if not on_memory:
            self.file = open(corpus_path, "r", encoding=encoding)
            self.random_file = open(corpus_path, "r", encoding=encoding)

            for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()
    

class LMDataset(BaseDataset):
    
    def __init__(self, corpus_path, vocab, seq_len, encoding="utf-8", corpus_lines=None, on_memory=True, n_corpus=None):
        super().__init__(corpus_path, vocab, seq_len, encoding, corpus_lines, on_memory)
        
        self.corpus = self.lines if on_memory else self.file
        if n_corpus is not None and n_corpus != corpus_lines:
            self.corpus = utils.reservoir_sampling(self.corpus, n_corpus)
            self.corpus_lines = self.n_corpus
        self.corpus = data_preprocess.normalize_sents(self.corpus)
        self.corpus = self.word_2_indices(self.corpus, vocab)
        
    
    def word_2_indices(corpus, vocab):

        indices_corpus = list()
        for line in corpus:
            indices_line = list()
            for char in line:
                char_index = vocab.stoi.get(char, vocab.unk_index)
                indices_line.append(char_index)
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
    
    data_dir = ""
    lang = ""
    filename = f"{lang}.corpus"
    
    corpus = open(os.path.join(data_dir, filename)).readlines() # lines = > a part of sample with ?
    
    seq_len_limit = 128
    
    
    
    