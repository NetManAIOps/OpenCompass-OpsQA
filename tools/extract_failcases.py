import json
import os
import re

import pandas as pd


def extract_answer(text: str) -> str:
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


def replace_bookname():
    books_df = pd.read_csv(
        '/mnt/mfs/opsgpt/evaluation/ops-cert-eval/books.csv')
    isbn_to_title = dict(
        zip(books_df['id'].tolist(), books_df['name'].tolist()))
    summary_df = pd.read_csv(
        '/mnt/mfs/opsgpt/opencompass/outputs/gpt/20230815_095657/summary/summary_20230815_095657.csv'  # noqa
    )
    summary_df['dataset'] = summary_df['dataset'].replace(isbn_to_title)
    summary_df.to_csv('/mnt/mfs/opsgpt/opencompass/outputs/gpt_summary.csv')


def main():

    folder = '/mnt/mfs/opsgpt/opencompass/outputs/gpt/20230815_095657/predictions/GPT-3.5-turbo'  # noqa

    failcases = []
    for file in os.listdir(folder):
        with open(os.path.join(folder, file), 'r', encoding='utf-8') as f:
            data = json.load(f)
        for idx, case in data.items():
            predict = extract_answer(case['prediction'])
            if predict != case['reference']['answer']:
                failcases.append(case)
                print(case['reference']['id'])
    with open('/mnt/mfs/opsgpt/opencompass/outputs/gpt3d5-failed-cases.json',
              'w') as f:
        json.dump(failcases, f, indent=4)


if __name__ == '__main__':
    replace_bookname()
