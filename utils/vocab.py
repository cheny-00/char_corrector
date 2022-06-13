



class Vocab:
    
    def __init__(self, vocab_path):
        
        self.vocab_path = vocab_path
        
        self.pad_index = 0
        self.mask_index = 1
        self.unk_index = 2
        
        special_tokens = ['[PAD]', '[MASK]', '[UNK]']
        
        self.itos = list(special_tokens)
        self.itos.extend([l.split('\t')[0] for l in open(vocab_path, 'r').readlines()])
        
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

        
        
        
        
        