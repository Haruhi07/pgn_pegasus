from tqdm import tqdm
import json
import torch
import argparse
import evaluate
from utils import set_random_seed
from pathlib import Path
from Model import PointerPegasus
from torch.utils.data import DataLoader, SequentialSampler
from datasets import load_from_disk
from transformers import PegasusTokenizer

seed = 42
set_random_seed(42)

results_dir = Path("./results")


def eval(model, tokenizer, eval_loader, device):
    rouge = evaluate.load("rouge")
    references = []
    predictions_ids = []
    model.eval()
    step = 0
    pbar = tqdm(total=len(eval_loader))

    for sample in eval_loader:
        step += 1
        pbar.update(1)

        references.append(sample['highlights'][0])

        input_ids = torch.LongTensor(sample["input_ids"]).to(device)
        with torch.no_grad():
            output = model.generate(input_ids=input_ids,
                                    max_new_tokens=128)[0]
        predictions_ids.append(output)
        #break
    predictions = tokenizer.batch_decode(predictions_ids, skip_special_tokens=True)
    print("-------------------Computing ROUGE Score-------------------")
    results = rouge.compute(predictions=predictions, references=references)

    with open(results_dir/"result.json", "w") as fp:
        json.dump(results, fp)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = 'google/pegasus-cnn_dailymail'
    tokenizer = PegasusTokenizer.from_pretrained(checkpoint)
    model = PointerPegasus(checkpoint, tokenizer, device).to(device)
    if args.model is not None:
        model.load_state_dict(args.model).to(device)

    tokenized_test = load_from_disk(args.dataset)
    test_loader = DataLoader(tokenized_test,
                             sampler=SequentialSampler(tokenized_test),
                             num_workers=2,
                             batch_size=1)

    eval(model, tokenizer, test_loader, device)