import json, re, os, codecs
from os import path as osp
from tools.utils import extract_questions
from tqdm import tqdm


def load_json(file):
    with open(file, 'r') as f:
        return json.load(f)

if __name__ == '__main__':
    translated = load_json("/mnt/mfs/opsgpt/evaluation/translated/translated_questions_v5.json")
    single = [q for q in translated if 'A' in q['id']]
    multiple = [q for q in translated if 'B' in q['id']]
    assert len(single) + len(multiple) == len(translated)


    with open("/mnt/mfs/opsgpt/evaluation/translated/v3/ch_single_test.json", 'w') as f:
        json.dump(single, f, indent=4, ensure_ascii=False)
    with open("/mnt/mfs/opsgpt/evaluation/translated/v3/ch_multiple_test.json", 'w') as f:
        json.dump(multiple, f, indent=4, ensure_ascii=False)