import os

import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset

import utils
from . import data_preprocess
from .vocab import Vocab

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
        self.corpus_lines = corpus_lines if corpus_lines is not None else len(corpus)
        self.corpus = corpus
        self.encoding = encoding
        

    

class LMDataset(BaseDataset):
    
    def __init__(self, corpus, vocab, seq_len, encoding="utf-8", corpus_lines=None, on_memory=True, n_corpus=None, max_len=512, load_half=False):
        super().__init__(corpus, vocab, seq_len, encoding, corpus_lines, on_memory)
        
        if load_half:
            self.corpus = corpus
            self.corpus_lines = len(corpus)
        else:
            if n_corpus is not None and n_corpus != corpus_lines:
                self.corpus = utils.reservoir_sampling(self.corpus, n_corpus)
                self.corpus_lines = n_corpus
            self.corpus = data_preprocess.normalize_sents(self.corpus)
            self.corpus = self.word_2_indices(self.corpus, vocab)
            self.corpus = self.format_corpus(self.corpus)
        self.max_len = max_len
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
                    print("unk char: ",char)
                indices_line.append(char_index)
            indices_corpus.append(indices_line)
        return indices_corpus
                
    
    def __len__(self):
        return self.corpus_lines
    
    def patch_data(self, corpus, idx):
        
        features = dict()
        line, line_len = self.format_line(corpus[idx].as_py(), self.max_len, self.vocab.pad_index, self.vocab.stoi['<eol>'])
        features['x'] = torch.tensor(line)
        features['x_len'] = line_len
        return features
    
    def format_corpus(self, data, *args, **kwargs):
        
        new_data = list()
        for line in data:
            new_data.append(self.format_line(line, *args, **kwargs))
        return new_data
            
    
    @staticmethod
    def format_line(line, max_len, pad_idx, eol_idx):
        if len(line) > max_len:
            line = line[len(line) - max_len:]
        line_len = len(line)
        if len(line) < max_len:
            line = [pad_idx] * (max_len - len(line)) + line
        line.append(eol_idx)
        
        return line, line_len
            
        
    
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
    
    
    
    