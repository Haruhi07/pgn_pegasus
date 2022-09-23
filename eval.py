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


def generate_reference(eval_dataset):
    print("-------------------Generating Reference-------------------")
    references = eval_dataset[:]["highlights"]
    with open(results_dir/"references.json", "w") as fp:
        json.dump(references, fp)


def generate_output(model, tokenizer, eval_loader):
    model.eval()
    pbar = tqdm(total=len(tokenized_test))
    step = 0
    predictions_ids = []

    for sample in eval_loader:
        step += 1
        pbar.update(1)

        input_ids = torch.tensor(sample["input_ids"]).to(device)
        with torch.no_grad():
            output = model.generate(input_ids=input_ids,
                                    max_new_tokens=128)[0]
        predictions_ids.append(output)

    predictions = tokenizer.batch_decode(predictions_ids, skip_special_tokens=True)
    with open(results_dir/"predictions.json", "w") as fp:
        json.dump(predictions, fp)


def eval(ref_path, pred_path):
    with open(ref_path, "r") as fp:
        references = json.load(fp)
    with open(pred_path, "r") as fp:
        predictions = json.load(fp)
    print("-------------------Computing ROUGE Score-------------------")
    rouge = evaluate.load("rouge")
    results = rouge.compute(predictions=predictions, references=references)
    print(results)
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
    generate_reference(eval_dataset=tokenized_test)
    test_loader = DataLoader(tokenized_test,
                             sampler=SequentialSampler(tokenized_test),
                             num_workers=2,
                             batch_size=1)

    generate_output(model=model,
                    tokenizer=tokenizer,
                    eval_loader=test_loader)

    eval_result = eval(ref_path=results_dir / "references.json",
                       pred_path=results_dir / "predictions.json")
    with open(results_dir / "random.json", "w") as fp:
        json.dump(eval_result, fp)