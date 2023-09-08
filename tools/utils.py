import json
import os
from os import path as osp

from opencompass.datasets import oreilly_choice_postprocess as extract_answer


def extract_questions(
        id_list: list[str],
        all_file: str = '/mnt/mfs/opsgpt/evaluation/ops-cert-eval/v3/all.json'
):
    """从题库中获得指定id的题目列表.

    Args:
        id_list (list[str]): 题目id列表
    """
    with open(all_file, encoding='utf-8', mode='r') as f:
        data = json.load(f)
    return [q for q in data if q['id'] in id_list]


def get_id_list_from_result_file(filename: str):
    """从一个输出文件中获得题目的id列表.

    Args:
        filename (str): 输出文件，格式如下：
        [{
            "origin_prompt": "",
            "prediction": "A" / ["A", "B"],
            "reference": [
                "id": "897xxxx",
                "answer": "B"
            ]
        }]

    Returns:
        _type_: _description_
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    return [q['reference']['id'] for q in data.values()]


def get_id_from_result_folder(folder: str):
    id_list = []
    for json_file in os.listdir(folder):
        if not osp.isfile(json_file) or '.json' not in json_file:
            continue
        id_list += get_id_list_from_result_file(osp.join(folder, json_file))
    return list(set(id_list))


def extract_failcases_from_result_file(result_file):
    failcases = []
    with open(os.path.join(result_file), 'r', encoding='utf-8') as f:
        data = json.load(f)
    for idx, case in data.items():
        predict = extract_answer(case['prediction'])
        gold = case['reference']['answer']
        if predict != gold:
            failcases.append(case)
    return failcases
