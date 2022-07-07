# 
import io
import gc
import gzip
import multiprocessing
from gensim.models import Word2Vec

import data_preprocess
from vocab import Vocab
from construct_dataset import BaseDataset


class EmbDataset(BaseDataset):
    
    def __init__(self, corpus, vocab, seq_len, encoding="utf-8", corpus_lines=None, on_memory=True, n_corpus=None, max_len=512):
        super().__init__(corpus, vocab, seq_len, encoding, corpus_lines, on_memory)
        
        self.corpus = data_preprocess.normalize_sents(self.corpus)
        # self.ids_corpus = self.word_2_indices(self.corpus, vocab)
        # self.save_indices(self.corpus, vocab)
    
    @staticmethod
    def corpus_to_indices(corpus, vocab):

        indices_corpus = list()
        for line in corpus:
            indices_line = list()
            line = line.split(' ')
            indices_line = line_to_idx(line, vocab)
            # yield indices_line
            indices_corpus.append(indices_line)
        return indices_corpus
    
            
                
    def __iter__(self):
        return self.ids_corpus
    
    def __len__(self):
        return self.corpus_lines
    
    @staticmethod
    def save_indices(save_corpus_path, corpus, vocab):
        
        with gzip.open(save_corpus_path+".gz", 'w') as f:
            with io.TextIOWrapper(f, encoding='utf-8') as dec:
                for line in tqdm(corpus, desc='comine line'):
                    indices_line = list()
                    line = line.split(' ')
                    for char in line:
                        char_index = vocab.stoi.get(char, vocab.unk_index)
                        if char_index == vocab.pad_index:
                            print("unk char: ",char)
                        indices_line.append(char_index)
                    dec.write(" ".join([str(_line) for _line in indices_line])+'\n')
                    
def line_to_idx(line, vocab):
    indices_line = list()
    for char in line:
        char_index = vocab.stoi.get(char, vocab.unk_index)
        if char_index == vocab.pad_index:
            print("unk char: ",char)
        indices_line.append(char_index)
    return indices_line
        
def load_indices(corpus_path):
    with gzip.open(corpus_path) as f:
        f_content = f.read()
    corpus = f_content.split(b'\n')
    del f_content
    gc.collect()
    corpus = [list(map(int, line.split(b' '))) for line in corpus]
    gc.collect()
    return corpus

def save_indices_wrapper(corpus, save_path, wrapper):
    import pyarrow as pa
    import pyarrow.parquet as pq
    data = data_preprocess.normalize_sents(corpus, wrapper)
    data_scheme = pa.schema({'idx': pa._list(pa.int32())})
    data_table = pa.table({'idx': data}, scheme=data_scheme)
    pq.write_table(data_table, save_path)

    

if __name__ == '__main__':
    
    from functools import partial
    
    import datasets
    from tqdm import tqdm
    
    vocab_path = '../data/en_dictionary.txt'
    emb_save_path = '../data/en_char_emb.wv'
    corpus_path = '/mnt/data/corpus_dataset/corpus/en/wikipedia/'
    corpus_data = datasets.load_from_disk(corpus_path)
    corpus_data = corpus_data.data['train']['text']
    n_len = len(corpus_data)
    save_corpus_path = lambda x: f'/mnt/data/corpus_dataset/indices_data/en_indices_{x}.corpus'
    
    train_data = list()
    # for i in tqdm((1, 2), desc='load data'):
    #     train_data.extend(load_indices(save_corpus_path(i)))
        
    corpus_vocab = Vocab(vocab_path)
    
    indices_wrapper = partial(line_to_idx, vocab=corpus_vocab)
    save_corpus_path(corpus_data, indices_wrapper, save_corpus_path('all'))
    exit()
    
    
    w2v_model = Word2Vec(train_data, vector_size=64, window=7, min_count=50,
                            sg=1, epochs=5 ,workers=multiprocessing.cpu_count())
    w2v_model.wv.save_word2vec_format(emb_save_path, binary=False) 
    # load wv2 file https://stackoverflow.com/questions/49710537/pytorch-gensim-how-to-load-pre-trained-word-embeddings/49802495#49802495
    # w2v_model = gensim.models.KeyedVectors.load_word2vec_format('path/to/file')
    # w2v_weight = w2v_model.wv
    
    # Trans to npy
    
    