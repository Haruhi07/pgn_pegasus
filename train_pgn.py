import argparse
import os
import sys
import json
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
from generate_dataset import PegasusDataset
from Model import PointerPegasus
from validation import validation
from transformers import PegasusConfig, PegasusTokenizer
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler


def train(model, device, tokenizer, train_loader, val_loader, sample_every=5000, grad_acc_steps=8, max_grad_norm=1.0):
    total_t0 = time.time()
    best_val_loss = None
    for epoch_i in range(1, epochs):
        # ========================================
        #               Training
        # ========================================
        print('======== Epoch {:} / {:} ========'.format(epoch_i, epochs))
        print('Training...')

        t0 = time.time()
        model.zero_grad()
        global_step = 0
        tr_loss = 0
        avg_train_loss = 0

        # train the pgn only
        model.eval()
        model.Pointer.train()
        for p in model.pegasus.parameters():
            p.requires_grad = False
        for p in model.Pointer.parameters():
            p.requires_grad = True

        for step, sample in enumerate(tqdm(train_loader)):
            input_ids = sample["input_ids"].to(device)
            attention_mask = sample["attention_mask"].to(device)
            labels = sample["decoder_input_ids"].to(device)

            output = model.forward(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   decoder_input_ids=labels)
            gen_probs, final_probs = output
            
            final_probs = final_probs.contiguous()
            labels = labels.contiguous()

            loss = loss_fct(final_probs.view(-1, final_probs.size(-1)), labels.view(-1))
            loss = loss/grad_acc_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.Pointer.parameters(), max_grad_norm)
            tr_loss += loss.item()
            
            if (step % grad_acc_steps == 0) or (step == len(tokenized_validation)):
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                optimizer.zero_grad()
                # 1 global_step = 1 gradient accumulation process
                global_step += 1
                # Calculate the average loss over all of the batches.
                avg_train_loss = tr_loss / global_step
                
                if global_step % sample_every == 0:
                    model.eval()
                    with torch.no_grad():
                        sample_tgt = model.generate(input_ids=input_ids, max_new_tokens=128)[0]
                        # Generate 1 sample every 'grad_acc_steps'*'sample_every' steps
                        print("{} {}".format(len(sample_tgt), tokenizer.decode(sample_tgt)), end='\n\n')
                        print("lr : {} tr_loss: {}".format(scheduler.get_last_lr()[0], avg_train_loss), end='\n\n')

        # Evaluate the model after each epoch
        avg_val_loss, validation_time = validation(model, device, val_loader, loss_fct)
        print("lr : {} tr_loss: {} val_loss: {}".format(scheduler.get_last_lr()[0], avg_train_loss, avg_val_loss), end='\n\n')
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        if (best_val_loss is None) or (avg_val_loss < best_val_loss):
            best_val_loss = avg_val_loss
            checkpoint = "epoch_{}.pt".format(epoch_i)
            torch.save(model, save_dir/checkpoint)
            tokenizer.save_pretrained(save_dir)
            print("model saved on epoch {} with val loss: {}", epoch_i, best_val_loss)

        print("")
        print("-------------------Average training loss: {0:.2f}-------------------".format(avg_train_loss))
        print("-------------------Training epoch took: {:}-------------------".format(training_time))

    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))


if __name__ == "__main__":
    seed = 42
    utils.set_random_seed(seed)

    save_dir = Path("./checkpoints")
    tmp_dir = Path('./tmp/tmpmodel.pt')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    print("-------------------Loading Data-------------------")
    dataset_dir = Path("./dataset_cache")
    with open(dataset_dir / "tokenized_training.json", 'rb') as fp:
        tokenized_training = pickle.load(fp)
    with open(dataset_dir / "tokenized_validation.json", 'rb') as fp:
        tokenized_validation = pickle.load(fp)

    train_loader = DataLoader(tokenized_training,
                              sampler=RandomSampler(tokenized_training),
                              num_workers=0,
                              batch_size=1)
    val_loader = DataLoader(tokenized_validation,
                            sampler=SequentialSampler(tokenized_validation),
                            num_workers=0,
                            batch_size=1)

    print("-------------------Data Loaded-------------------")

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=7)
    parser.add_argument("--grad_acc_steps", default=8)
    parser.add_argument("--sample_every", default=10000)
    parser.add_argument("--checkpoint", default="google/pegasus-cnn_dailymail")
    args = parser.parse_args()

    epochs = args.epochs
    learning_rate = 5e-5
    warmup_ratio = 0.1
    epsilon = 1e-8
    grad_acc_steps = args.grad_acc_steps
    total_steps = len(tokenized_training)
    sample_every = args.sample_every
    checkpoint = args.checkpoint
    startpoint = "google/pegasus-cnn_dailymail"

    print("-------------------Creating A Instance for the Model-------------------")
    tokenizer = PegasusTokenizer.from_pretrained(startpoint)
    if checkpoint == startpoint:
        model = PointerPegasus(checkpoint, tokenizer, device).to(device)
    else:
        print("loading model from {}".format(checkpoint))
        model = torch.load(checkpoint).to(device)

    print("-------------------Instance Created-------------------")

    optimizer = AdamW(model.Pointer.parameters(),
                      lr=learning_rate,
                      eps=epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_ratio * total_steps,
                                                num_training_steps=total_steps)
    loss_fct = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    train(model,
          device,
          tokenizer,
          train_loader=train_loader,
          val_loader=val_loader,
          sample_every=sample_every,
          grad_acc_steps=grad_acc_steps)