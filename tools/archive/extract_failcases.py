import json  # noqa
import os  # noqa
import re  # noqa
from os import path as osp  # noqa

import pandas as pd  # noqa

from opencompass.datasets import oreilly_choice_postprocess as extract_answer


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


def extract_fail_cases(folder):
    os.makedirs(osp.join(folder, 'failcases'), exist_ok=True)
    for model in os.listdir(osp.join(folder, 'predictions')):
        print(model)

        for filename in os.listdir(osp.join(folder, 'predictions', model)):
            failcases = []
            if not osp.isfile(osp.join(folder, 'predictions', model,
                                       filename)):
                continue
            with open(os.path.join(folder, 'predictions', model, filename),
                      'r',
                      encoding='utf-8') as f:
                data = json.load(f)
            for idx, case in data.items():
                predict = extract_answer(case['prediction'])
                gold = case['reference']['answer']
                if predict != gold:
                    failcases.append(case)
            with open(
                    osp.join(folder, 'failcases',
                             f'{model}_{filename.replace(".json", "")}.json'),
                    'w') as f:
                json.dump(failcases, f, indent=4)


if __name__ == '__main__':
    workdir = '/mnt/mfs/opsgpt/opencompass/outputs/'
    # extract_fail_cases(workdir + 'archive/internxverse-oreilly-sample')
    # extract_fail_cases(workdir + 'archive/baichuan-oreilly-sample')
    # extract_fail_cases(workdir + 'archive/glmllama-oreilly-sample')
    # extract_fail_cases(workdir + 'archive/gpt-oreilly-sample')
    # extract_fail_cases(workdir + 'archive/llama2chat13b-oreilly-full')
    # extract_fail_cases(workdir + 'archive/gpt3.5-oreilly-full')
    workdir = '/mnt/mfs/opsgpt/opencompass/archive/v-xll22/outputs/'
    # extract_fail_cases(workdir + 'few_shot/chatglm/20230818_021329')
    # extract_fail_cases(workdir + 'few_shot/chatgpt/20230818_001957')
    # extract_fail_cases(workdir + 'few_shot/internlm/20230817_230433')
    extract_fail_cases(workdir + 'few_shot/qwen/20230818_023645')
    # extract_fail_cases(workdir + 'sc/chatglm-6b/20230816_181124')
    # extract_fail_cases(workdir + 'zero_shot/chatgpt/20230818_023958')
    # extract_fail_cases(workdir + 'zero_shot/chinese-alpaca-2/20230817_013153')  # noqa
    # extract_fail_cases(workdir + 'zero_shot/chineses-llama-2-13b/20230816_195746')  # noqa
    # extract_fail_cases(workdir + 'zero_shot/qwen/20230817_163308')
