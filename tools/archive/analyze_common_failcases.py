import csv  # noqa
import json
import os
import re  # noqa
import sys  # noqa
from os import path as osp

import pandas as pd  # noqa

if __name__ == '__main__':
    folder = '/mnt/mfs/opsgpt/opencompass/outputs/failcases/en_zh_comparisons/all'  # noqa
    fail_ids = dict()
    for filename in os.listdir(folder):
        with open(osp.join(folder, filename), 'r') as f:
            data = json.load(f)
        fail_ids[filename] = set([d['reference']['id'] for d in data])

    intersect = None
    for key, value in fail_ids.items():
        print(key, len(value))
        if not intersect:
            intersect = value
        else:
            intersect = intersect & value

    print(len(intersect))
    failcases = dict()
    # get question and answer
    with open(
            osp.join(folder,
                     'baichuan-7b_zh-en-comparison-zeroshot-sc-zh.json'),
            'r') as f:
        data = json.load(f)
        for d in data:
            if d['reference']['id'] in intersect:
                failcases[d['reference']['id']] = {
                    'zh-prompt': d['origin_prompt'],
                    'answer': d['reference']['answer']
                }
    with open(
            osp.join(folder,
                     'baichuan-7b_zh-en-comparison-zeroshot-sc-en.json'),
            'r') as f:
        data = json.load(f)
        for d in data:
            if d['reference']['id'] in intersect:
                failcases[d['reference']
                          ['id']]['en-prompt'] = d['origin_prompt']
    # get each question's multiple answers
    for filename in os.listdir(folder):
        with open(osp.join(folder, filename), 'r') as f:
            data = json.load(f)
        for d in data:
            if d['reference']['id'] in intersect:
                failcases[d['reference']['id']][filename.replace(
                    '.json', '')] = d['prediction']

    with open(osp.join(folder, '../merged.json'), 'w') as f:
        json.dump(failcases, f, indent=4)

    acnt = 0
    bcnt = 0
    for wid in intersect:
        if 'A' in wid:
            acnt += 1
        else:
            bcnt += 1
    print(acnt, bcnt)
