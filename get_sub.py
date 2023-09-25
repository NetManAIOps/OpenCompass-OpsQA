import os
import json
import random
import argparse
import evaluate
import numpy as np


def gen_sub(input_path, out_path, sc=False):
    seed = 233
    random_state = random.getstate()
    np_random_state = np.random.get_state()

    random.seed(seed)
    np.random.seed(seed)
    # metric = evaluate.load('sacrebleu')
    res_list = []
    files = os.listdir(input_path)
    files.sort()
    for file in files:
        with open(os.path.join(input_path, file), 'r') as f:
            res = json.load(f)
            res_list.extend(list(res.values()))
    print(f"All {len(res_list)} results.")
    predictions = [res['prediction'].strip() for res in res_list]
    keypoints = [res['reference']['keypoint'] for res in res_list]
    references = [res['reference']['keypoint'] + ' ' + res['reference']['answer'] for res in res_list]
    rouge = evaluate.load('rouge')
    rouge_score = rouge.compute(predictions=predictions, references=keypoints)
    rouge_score = {k: v * 100 for k, v in rouge_score.items()}
    bleu = evaluate.load('sacrebleu')
    bleu_score = bleu.compute(predictions=predictions, references=references)
    random.setstate(random_state)
    np.random.set_state(np_random_state)
    print(f"BLEU: {bleu_score}\nROUGE: {rouge_score}")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'BLEU': bleu_score, 'ROUGE': rouge_score}, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='directory of prediction files', type=str, required=True)
    parser.add_argument("--out", help="output path", type=str, required=True)
    parser.add_argument("--sc", help="use self-consistency or not", type=lambda x: x.lower() in ['true', '1'], default=False)
    args = parser.parse_args()
    gen_sub(args.input, args.out, args.sc)
    