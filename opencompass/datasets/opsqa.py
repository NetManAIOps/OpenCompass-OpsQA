import csv
import os.path as osp

from datasets import Dataset, DatasetDict  # noqa

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class OpsQADataset(BaseDataset):

    @staticmethod
    def load(path: str):
        raw_data = []
        filename = osp.join(path, 'demo_csv.csv')
        with open(filename, encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                assert len(row) == 6
                raw_data.append({
                    'input': row[0],
                    'A': row[1],
                    'B': row[2],
                    'C': row[3],
                    'D': row[4],
                    'target': row[5],
                })
        return Dataset.from_list(raw_data)
