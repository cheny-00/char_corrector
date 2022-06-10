
import multiprocessing
from gensim.models import Word2Vec

from utils.vocab import Vocab
from utils.construct_dataset import LMDataset

def train_word2vec(data):
    data_indices_str = []
    for d in data[:2]:
        for x in d:
            data_indices_str.append(list(map(str, x)))
    w2v_model = Word2Vec(data_indices_str, vector_size=50, window=7, min_count=50,
                            sg=1, epochs=5 ,workers=multiprocessing.cpu_count())
    return w2v_model
    
#embeddings.wv.save_word2vec_format(emb_path, binary=False)


if __name__ == '__main__':
    
    vocab_path = '../data/en_dictionary.txt'
    emb_save_path = '../data/en_char_emb.wv'
    corpus_path = ''
    
    
    corpus_vocab = Vocab(vocab_path)
    
    data = LMDataset(corpus_path)
    