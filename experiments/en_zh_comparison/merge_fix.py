import json, re, os, codecs
from os import path as osp
from tools.utils import extract_questions
from tqdm import tqdm


def load_json(file):
    with open(file, 'r') as f:
        return json.load(f)

if __name__ == '__main__':
    translated = load_json("/mnt/mfs/opsgpt/evaluation/translated/translated_questions_v2.json")
    fixed = load_json("/mnt/mfs/opsgpt/opencompass/experiments/en_zh_comparison/failed.json")
    fix_list = [q['id'] for q in fixed]
    merge = fixed + [q for q in translated if q['id'] not in fix_list]
    with open("/mnt/mfs/opsgpt/evaluation/translated/translated_questions_v5.json", 'w') as f:
        json.dump(merge, f, indent=4, ensure_ascii=False)