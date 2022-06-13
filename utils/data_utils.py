
import datasets


def load_train_data_from_datasets(data_path):
    train_data = datasets.load_from_disk

def load_raw_data_from_datasets(data_path):
    data = datasets.load_from_disk(data_path)
    return data