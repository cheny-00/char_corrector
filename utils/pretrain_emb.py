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
    line = line.split(" ")
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
    data_preprocess.for_save_data_wrapper(corpus, save_path, wrapper)

def load_pa_data(data_path):
    
    import pyarrow.parquet as pq
    data = pq.read_table(data_path)
    return data
    
def trans_to_pa_table():

    vocab_path = '../data/en_dictionary.txt'
    emb_save_path = '../data/en_char_emb.wv'
    corpus_path = '/mnt/data/corpus_dataset/corpus/en/wikipedia/'
    corpus_data = datasets.load_from_disk(corpus_path)
    # corpus_data = corpus_data.data['train']['text']
    n_len = len(corpus_data)
    save_corpus_path = lambda x: f'/mnt/data/corpus_dataset/indices_data/en_indices_{x}.arrow'
    
    part_corpus_data = lambda x, y: corpus_data.data['train'][x:y]['text']
    corpus_vocab = Vocab(vocab_path)
    indices_wrapper = partial(line_to_idx, vocab=corpus_vocab)
    train_data = list()
    n = len(corpus_data['train'])
    # n=100
    mid = n // 2
    save_indices_wrapper(corpus_data['train']['text'], save_corpus_path('test'), indices_wrapper)
    # for i, r in zip((1, 2), ((0, mid), (mid, n))):
    #     gc.collect()
    #     train_data += save_indices_wrapper(part_corpus_data(*r), wrapper=indices_wrapper, save_path=save_corpus_path(i))
    
class TrainData:
    def __init__(self, data):
        self.data =  data
    def __getitem__(self, i):
        return self.data[i]
    def __iter__(self):
        for _data in self.data:
            yield [str(_i) for _i in _data]
    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    
    from functools import partial
    
    import datasets
    from tqdm import tqdm
    import pyarrow as pa
    # trans_to_pa_table()
    # exit()
    save_corpus_path = lambda x: f'/mnt/data/corpus_dataset/indices_data/en_indices_{x}.arrow'
    emb_save_path = '../data/en_char_emb.wv'
    
    with pa.memory_map(save_corpus_path('all'), 'rb') as source:
        train_data = pa.ipc.open_file(source).read_all()
    n_words = 0 
    train_data = train_data['idx']
    # train_data = train_data['idx'][:100]
    train_data = TrainData(train_data)
    n_samples = len(train_data)
    print("numbers of data: ", n_samples)
    for _d in tqdm(train_data, desc='count numbers of words'):
        n_words += len(_d)
    print("n_words :", n_words)
    w2v_model = Word2Vec(vector_size=64, window=5, min_count=50, 
                         sg=1, epochs=2 ,workers=multiprocessing.cpu_count())
    w2v_model.build_vocab(tqdm(train_data), progress_per=n_words)
    print("build done ", w2v_model.corpus_count)
    w2v_model.train(tqdm(train_data), total_examples=n_samples, epochs=w2v_model.epochs)
    print(w2v_model.wv)
    w2v_model.wv.save_word2vec_format(emb_save_path, binary=False) 
    # load wv2 file https://stackoverflow.com/questions/49710537/pytorch-gensim-how-to-load-pre-trained-word-embeddings/49802495#49802495
    # w2v_model = gensim.models.KeyedVectors.load_word2vec_format('path/to/file')
    # w2v_weight = w2v_model.wv
    
    # Trans to npy
     