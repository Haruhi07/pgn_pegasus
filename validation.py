import argparse
import time
import torch
import pickle
import random
import numpy as np

import utils
from utils import set_random_seed, format_time
import transformers
from tqdm import tqdm
from pathlib import Path
from datasets import load_from_disk
from Model import PointerPegasus
from transformers import PegasusConfig, PegasusTokenizer
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler
from generate_dataset import PegasusDataset


def validation(model, val_loader, loss_fct):
    # ========================================
    #               Validation
    # ========================================
    print("")
    print("Running Validation...")

    t0 = time.time()

    model.eval()

    total_eval_loss = 0

    # Evaluate data for one epoch
    for step, sample in enumerate(tqdm(val_loader)):

        input_ids = sample["input_ids"].to(device)
        labels = sample["decoder_input_ids"].to(device)
        attention_mask = sample["attention_mask"].to(device)

        with torch.no_grad():
            output = model(input_ids=input_ids,
                           attention_mask=attention_mask,
                           decoder_input_ids=labels)
        gen_probs, final_probs = output

        final_probs = final_probs.contiguous()
        labels = labels.contiguous()

        loss = loss_fct(final_probs.view(-1, final_probs.size(-1)), labels.view(-1))

        total_eval_loss += loss.item()

    avg_val_loss = total_eval_loss / len(val_loader)
    validation_time = format_time(time.time() - t0)

    print("-------------------Validation Loss: {0:.5f}-------------------".format(avg_val_loss))
    print("-------------------Validation took: {:}-------------------".format(validation_time))

    return avg_val_loss, validation_time


if __name__ == "__main__":
    dataset_dir = Path("./dataset_cache")
    with open(dataset_dir / "tokenized_validation.json", 'rb') as fp:
        tokenized_validation = pickle.load(fp)
    val_loader = DataLoader(tokenized_validation,
                            sampler=SequentialSampler(tokenized_validation),
                            num_workers=2,
                            batch_size=1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = "google/pegasus-cnn_dailymail"
    tokenizer = PegasusTokenizer.from_pretrained(checkpoint)
    model = PointerPegasus(checkpoint, tokenizer, device).to(device)
    loss_fct = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    validation(model, val_loader, loss_fct)