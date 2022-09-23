import tqdm
import json
import torch
import evaluate
from pathlib import Path


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
