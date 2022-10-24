import json
import pickle
import torch
import argparse
import evaluate
from tqdm import tqdm
from utils import set_random_seed
from pathlib import Path
from Model import PointerPegasus
from torch.utils.data import DataLoader, SequentialSampler
from datasets import load_from_disk
from generate_dataset import PegasusDataset
from transformers import PegasusTokenizer

seed = 42
set_random_seed(42)

results_dir = Path("./results")


def generate_reference(eval_dataset):
    print("-------------------Generating Reference-------------------")
    references = eval_dataset[:]["highlights"]
    with open(results_dir/"references.json", "w") as fp:
        json.dump(references, fp)


def generate_output(model, tokenizer, eval_loader, save_dir):
    return
    model.eval()
    predictions_ids = []
    step = 0

    for sample in tqdm(eval_loader):

        input_ids = sample["input_ids"].to(device)
        with torch.no_grad():
            output = model.generate(input_ids=input_ids,
                                    max_new_tokens=128)[0]
        predictions_ids.append(output)

    predictions = tokenizer.batch_decode(predictions_ids, skip_special_tokens=True)
    with open(save_dir, "w") as fp:
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
    parser.add_argument("--checkpoint", default='google/pegasus-cnn_dailymail')
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--epoch", required=True)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    startpoint = 'google/pegasus-cnn_dailymail'
    tokenizer = PegasusTokenizer.from_pretrained(startpoint)
    if args.checkpoint != startpoint:
        print("loading model from {}".format(args.checkpoint))
        model = torch.load(args.checkpoint).to(device)
    else:
        model = PointerPegasus(startpoint, tokenizer, device).to(device)

    test_dir = Path(args.dataset)/"tokenized_test.json"
    print("loading tokenized_test...")
    with open(test_dir, 'rb') as fp:
        tokenized_test = pickle.load(fp)
    #generate_reference(eval_dataset=tokenized_test)
    test_loader = DataLoader(tokenized_test,
                             sampler=SequentialSampler(tokenized_test),
                             num_workers=0,
                             batch_size=1)

    generate_output(model=model,
                    tokenizer=tokenizer,
                    eval_loader=test_loader,
                    save_dir=results_dir / "epoch_{}".format(args.epoch) / "predictions.json")

    eval_result = eval(ref_path=results_dir / "references.json",
                       pred_path=results_dir / "epoch_{}".format(args.epoch) / "predictions.json")
    with open(results_dir / "epoch_{}".format(args.epoch) / "epoch_{}.json".format(args.epoch), "w") as fp:
        json.dump(eval_result, fp)