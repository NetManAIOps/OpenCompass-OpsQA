import os
import json
import re
import argparse
import pandas as pd
from collections import Counter

def oreilly_choice_postprocess(text: str) -> str:
    s = text
    s += ' '
    matched = re.search(r'answer.+?[^\w]([a-zA-Z][^\w])', s.lower())
    if matched:
        s = s[matched.span()[0]:]
    if not re.match(r'[A-Z][^\w]', s):
        matched = re.search(r'[^\w][A-Z][^\w]', s)
        if matched:
            s = s[matched.span()[0] + 1:]

    s = s.strip() + ' '
    pattern = r'([A-Z][\s,]+(and)?[\s,]*)*[A-Z][^\w]'
    matched = re.match(pattern, s, re.S)
    prefix = matched.group() if matched else ''
    ans = list(
        set([
            c.upper() for c in re.split('[^A-Z]', prefix)
            if len(c) == 1 and c.isalpha()
        ]))
    ans.sort()
    return ','.join(ans)


def gen_accuracy(input_path, out_path, sc=False, sample=False, sample_path=''):
    res_list = []
    for file in os.listdir(input_path):
        with open(os.path.join(input_path, file), 'r') as f:
            res = json.load(f)
            res_list.extend(list(res.values()))
    print(f"All {len(res_list)} results.")

    if sample:
        with open(sample_path, 'r') as f:
            sample_ids = json.load(f)

    book_all, book_correct, book_acc = {}, {}, {}
    correct_num = 0
    sample_num = 0
    for res in res_list:
        question_id = res['reference']['id']
        if sample and question_id not in sample_ids:
                continue
        if sample:
            sample_num += 1
        if sc:
            predictions = [oreilly_choice_postprocess(pred) for pred in res['prediction']]
            prediction = Counter(predictions).most_common(1)[0][0]
        else:
            prediction = oreilly_choice_postprocess(res['prediction'])
        answer = res['reference']['answer']
        book = question_id.split('-')[0]
        cor = 1 if prediction == answer else 0
        correct_num += cor
        if book in book_all:
            book_all[book] += 1
            book_correct[book] += cor
        else:
            book_all[book] = 1
            book_correct[book] = cor
    if sample:
        print(f"Using sample, all {sample_num} samples.")
    for key in book_all.keys():
        book_acc[key] = book_correct[key] / book_all[key] * 100
    all_acc = correct_num / len(res_list) * 100 if not sample else correct_num / sample_num * 100
    print(f"Accuracy: {all_acc}")
    sorted_book_acc = dict(sorted(book_acc.items(), key=lambda x: x[0]))
    for key in sorted_book_acc.keys():
        sorted_book_acc[key] = [round(sorted_book_acc[key], 4)]
    df = pd.DataFrame(sorted_book_acc)
    df.to_csv(out_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='directory of prediction files', type=str, required=True)
    parser.add_argument("--out", help="dataset path", type=str, required=True)
    parser.add_argument("--sc", help="use self-consistency or not", type=lambda x: x.lower() in ['true', '1'], default=False)
    parser.add_argument("--sample", help="use sample or not", type=lambda x: x.lower() in ['true', '1'], default=False)
    parser.add_argument("--sample_path", help="sample ids path", type=str, default='/home/v-xll22/OpsGPT/OpenCompass-OpsQA/sample_id.json')
    args = parser.parse_args()
    gen_accuracy(args.input, args.out, args.sc, args.sample, args.sample_path)
    