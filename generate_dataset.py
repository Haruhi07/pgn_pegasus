import sys
import pickle
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset
from transformers import PegasusTokenizer
from torch.utils.data.dataset import Dataset


class PegasusDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=128, max_source_length=512):
        self.dataset = []
        for idx, sample in enumerate(tqdm(dataset)):
            input_ids = tokenizer(sample["article"],
                                   truncation=True,
                                   max_length=max_source_length,
                                   return_tensors='pt')["input_ids"][0]
            decoder_input_ids = tokenizer(tokenizer.pad_token+sample["highlights"],
                                           truncation=True,
                                           max_length=max_length,
                                           return_tensors='pt')["input_ids"][0]
            self.dataset.append({"article": sample["article"],
                                 "highlights": sample["highlights"],
                                 "input_ids": input_ids,
                                 "decoder_input_ids": decoder_input_ids})

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    cache_dir = Path("./cache")
    dataset_dir = Path("./dataset_cache")
    tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-cnn_dailymail")

    sys.stderr.write("before loading")
    training = load_dataset("cnn_dailymail", "3.0.0", split="train", cache_dir=cache_dir)
    validation = load_dataset("cnn_dailymail", "3.0.0", split="validation", cache_dir=cache_dir)
    test = load_dataset("cnn_dailymail", "3.0.0", split="test", cache_dir=cache_dir)

    sys.stderr.write("after loading")
    #tokenized_training = PegasusDataset(training, tokenizer)
    tokenized_validation = PegasusDataset(validation, tokenizer)
    tokenized_test = PegasusDataset(test, tokenizer)

    #with open(dataset_dir/"tokenized_training.json", "wb") as fp:
        #pickle.dump(tokenized_training, fp)
    with open(dataset_dir/"tokenized_validation.json", "wb") as fp:
        pickle.dump(tokenized_validation, fp)
    with open(dataset_dir/"tokenized_test.json", "wb") as fp:
        pickle.dump(tokenized_test, fp)
