from pathlib import Path
from datasets import load_dataset
from transformers import PegasusTokenizer

cache_dir = Path("./cache")
dataset_dir = Path("./dataset_cache")
max_length=128
max_source_length=1024
tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-cnn_dailymail")


def tokenize_dataset(sample):
    input_dict = tokenizer(sample["article"],
                           truncation=True,
                           max_length=max_source_length,
                           return_tensors='pt')
    decoder_input_dict = tokenizer(tokenizer.pad_token + sample["highlights"],
                                   truncation=True,
                                   max_length=max_length,
                                   return_tensors='pt')
    sample["input_ids"] = input_dict["input_ids"]
    sample["attention_mask"] = input_dict["attention_mask"]
    sample["decoder_input_ids"] = decoder_input_dict["input_ids"]
    sample["decoder_attention_mask"] = decoder_input_dict["attention_mask"]
    return sample

training = load_dataset("cnn_dailymail", "3.0.0", split="train", cache_dir=cache_dir)
validation = load_dataset("cnn_dailymail", "3.0.0", split="validation", cache_dir=cache_dir)
test = load_dataset("cnn_dailymail", "3.0.0", split="test", cache_dir=cache_dir)

tokenized_training = training.map(tokenize_dataset)
tokenized_validation = validation.map(tokenize_dataset)
tokenized_test = test.map(tokenize_dataset)

tokenized_training.save_to_disk(dataset_dir)
tokenized_validation.save_to_disk(dataset_dir)
tokenized_test.save_to_disk(dataset_dir)
