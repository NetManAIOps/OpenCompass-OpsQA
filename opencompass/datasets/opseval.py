import csv  # noqa
import json
import os.path as osp  # noqa
import random
import re
from datasets import Dataset, DatasetDict  # noqa
from opencompass.openicl.icl_evaluator import (BaseEvaluator, BleuEvaluator,
                                               RougeEvaluator)
from opencompass.registry import TEXT_POSTPROCESSORS  # noqa
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET

from .base import BaseDataset

@staticmethod
def apply_sample_setting(sample_setting, split, json_data):
    # Sampling settings
    sample_ids = None
    if sample_setting and split == 'test':
        if 'seed' in sample_setting:
            random.seed(sample_setting['seed'])
        else:
            random.seed(0)
        if 'load_list' in sample_setting:
            with open(sample_setting['load_list'],
                    'r',
                    encoding='utf-8') as f:
                sample_ids = json.load(f)
        elif 'sample_size' in sample_setting:
            sample_size = sample_setting['sample_size']
            sample_size = min(sample_size, len(json_data))
            sample_ids = random.sample(list(range(len(json_data))),
                                    sample_size)
        elif 'sample_frac' in sample_setting:
            sample_frac = sample_setting['sample_frac']
            assert sample_frac <= 1, 'Sample Frac must be equal or lesser to 1!'  # noqa: E501
            sample_len = int(len(json_data) * sample_frac)
            sample_len = 1 if sample_len == 0 else sample_len
            sample_ids = random.sample(list(range(len(json_data))),
                                    sample_len)
    if 'load_list' in sample_setting and sample_ids and split == 'test':  # noqa
        json_data = [
            item for item in json_data if item['id'] in sample_ids
        ]
    elif sample_ids and split == 'test':
        json_data = [
            item for idx, item in enumerate(json_data)
            if idx in sample_ids
        ]
    return json_data


@LOAD_DATASET.register_module()
class OpsEvalMCDataset(BaseDataset):
    """
    Dataset Class for OpsEval Multi-Choice Questions
    fields:
        - question: str required
        - answer: str required
        - A/B/C/D: str required
        - solution: str optional, required for CoT
        - id: str required
    """
    @staticmethod
    def load(path: str, name: str, sample_setting: dict = None):
        dataset = DatasetDict()
        for split in ['dev', 'test']:
            with open(osp.join(path, f'{name}_{split}.json'),
                      encoding='utf-8') as f:
                json_data = json.load(f)

            json_data = apply_sample_setting(sample_setting, split, json_data)                
            print("[DATASET DEBUG]: dataset size", len(json_data))

            raw_data = []
            for data in json_data:
                try:
                    item = {
                        'question': data['question'],
                        'answer': data['answer'],
                        'A': data['A'],
                        'B': data['B'],
                        'C': data['C'] if 'C' in data else '',
                        'D': data['D'] if 'D' in data else '',
                        'E': data['E'] if 'E' in data else '',
                        'F': data['E'] if 'F' in data else '',
                        'choices': data['choices'] if 'choices' in data else [],
                        'solution': data['solution'] if 'solution' in data else '',
                        'id': data['id'],
                    }
                    # 检查E到H是否存在，如果存在则加入，（但是模板里我们不会考虑这种情况）
                    for i in range(4, 8):
                        if chr(65 + i) in data:
                            item[chr(65 + i)] = data[chr(65 + i)]

                    raw_data.append(item)
                except KeyError as err:
                    print("[DATASET DEBUG]: data", data)
                    raise err
            dataset[split] = Dataset.from_list(raw_data)
        return dataset


@LOAD_DATASET.register_module()
class OpsEvalQADataset(BaseDataset):
    """
    Dataset Class for OpsEval Question-Answering Questions
    fields:
        - question: str required
        - answer: str required
        - solution: str optional
        - id: str required
    """
    @staticmethod
    def load(path: str, name: str, sample_setting: dict = None):
        dataset = DatasetDict()
        for split in ['dev', 'test']:
            with open(osp.join(path, f'{name}_{split}.json'),
                      encoding='utf-8') as f:
                json_data = json.load(f)

            json_data = apply_sample_setting(sample_setting, split, json_data)                
            print("[DATASET DEBUG]: dataset size", len(json_data))

            raw_data = []
            for data in json_data:
                item = {
                    'question': data['question'],
                    'answer': data['answer'],
                    'solution': data['solution'] if 'solution' in data else '',
                    'id': data['id'],
                }
                raw_data.append(item)
            dataset[split] = Dataset.from_list(raw_data)
        return dataset