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

@LOAD_DATASET.register_module()
class QADataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str, sample_setting: dict = None):

        dataset = DatasetDict()
        for split in ['dev', 'test']:
            with open(osp.join(path, f'{name}_{split}.json'),
                      encoding='utf-8') as f:
                json_data = json.load(f)

            # Sampling settings
            sample_ids = None
            if sample_setting:
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
            if 'load_list' in sample_setting and sample_ids:  # noqa
                json_data = [
                    item for item in json_data if item['id'] in sample_ids
                ]
            elif sample_ids:
                json_data = [
                    item for idx, item in enumerate(json_data)
                    if idx in sample_ids
                ]
                
            print("[DATASET DEBUG]: dataset size", len(json_data))

            raw_data = []
            for data in json_data:
                item = {
                    'question': data['question'],
                    'answer': data['answer'],
                    # 'solution': data['solution'],
                    'id': data['id'],
                }
                raw_data.append(item)
            dataset[split] = Dataset.from_list(raw_data)
        return dataset

@LOAD_DATASET.register_module()
class QAZeroshotDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str, sample_setting: dict = None):

        dataset = DatasetDict()
        with open(osp.join(path, f'{name}.json'),
                    encoding='utf-8') as f:
            json_data = json.load(f)

        # Sampling settings
        sample_ids = None
        if sample_setting:
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
        if 'load_list' in sample_setting and sample_ids:  # noqa
            json_data = [
                item for item in json_data if item['id'] in sample_ids
            ]
        elif sample_ids:
            json_data = [
                item for idx, item in enumerate(json_data)
                if idx in sample_ids
            ]
            
        print("[DATASET DEBUG]: dataset size", len(json_data))

        raw_data = []
        for data in json_data:
            item = {
                'question': data['question'],
                'answer': data['answer'],
                # 'solution': data['solution'],
                'id': data['id'],
            }
            raw_data.append(item)
        dataset['test'] = Dataset.from_list(raw_data)
        dataset['dev'] = Dataset.from_list([])
        return dataset