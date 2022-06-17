import datasets

class BaseCollate:
    def __init__(self, corpus) -> None:
        self.corpus = corpus.data
        
class TextCollate(BaseCollate):
    def __init__(self, corpus) -> None:
        super().__init__(corpus)
    def __len__(self):
        return self.corpus.num_rows
    def __getitem__(self, idx):
        return self.corpus['text'][idx]
    
    def __iter__(self):
        return self.corpus

def data_collate(dataset_path, dataset_name):
    collate_table = {
        "bookcorpus": TextCollate,
    }
    corpus_data = datasets.load_from_disk(dataset_path)
    return collate_table[dataset_name](corpus_data)
    
    