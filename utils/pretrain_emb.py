
import multiprocessing
from gensim.models import Word2Vec

from vocab import Vocab
from construct_dataset import LMDataset





if __name__ == '__main__':
    
    import datasets
    
    vocab_path = '../data/en_dictionary.txt'
    emb_save_path = '../data/en_char_emb.wv'
    corpus_path = '/mnt/d/datasets/en/bookcorpus/saved/train'
    corpus_data = datasets.load_from_disk(corpus_path)
    corpus_data = corpus_data[:100]['text']
    
    corpus_vocab = Vocab(vocab_path)
    
    train_data = LMDataset(corpus=corpus_data, vocab=corpus_vocab, seq_len=100, n_corpus=1000)
    
    w2v_model = Word2Vec(train_data.corpus, vector_size=64, window=7, min_count=50,
                            sg=1, epochs=5 ,workers=multiprocessing.cpu_count())
    w2v_model.wv.save_word2vec_format(emb_save_path, binary=False) 
    # load wv2 file https://stackoverflow.com/questions/49710537/pytorch-gensim-how-to-load-pre-trained-word-embeddings/49802495#49802495
    # w2v_model = gensim.models.KeyedVectors.load_word2vec_format('path/to/file')
    # w2v_weight = w2v_model.wv
    
    # Trans to npy
    
    