import csv
import json
import os.path as osp  # noqa

from datasets import Dataset, DatasetDict  # noqa

from opencompass.openicl.icl_evaluator import (BaseEvaluator, BleuEvaluator,
                                               RougeEvaluator)
from opencompass.registry import (ICL_EVALUATORS, LOAD_DATASET,
                                  TEXT_POSTPROCESSORS)

from .base import BaseDataset


@LOAD_DATASET.register_module()
class OpsQAChoiceDemoDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        raw_data = []
        with open(path, encoding='utf-8') as f:
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


@LOAD_DATASET.register_module()
class OpsQASummaryDemoDataset(BaseDataset):
    """
    摘要生成 Demo
    字段：
    context
    hint
    answer
    """

    @staticmethod
    def load(path: str):
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        return Dataset.from_list(data)


@LOAD_DATASET.register_module()
class OpsQAComprDemoDataset(BaseDataset):
    """Comprehenshion Demo 参考了opencompass/datasets/cmrc.py CMRCDataset."""

    @staticmethod
    def load(path: str):
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        # 将原始数据转换为所需的格式
        rows = []
        for index, paragraph in enumerate(data):
            context = paragraph['context']
            for question in paragraph['qas']:
                answers = question['answers']
                unique_answers = list(set([a['text'] for a in answers]))
                rows.append({
                    'context': context,
                    'question': question['question'],
                    'answers': unique_answers
                })
        # 创建 Dataset
        dataset = Dataset.from_dict({
            'context': [row['context'] for row in rows],
            'question': [row['question'] for row in rows],
            'answers': [row['answers'] for row in rows]
        })

        return dataset


@TEXT_POSTPROCESSORS.register_module('opsqa_compr')
def opsqa_compr_postprocess(text: str) -> str:
    if '答案是' in text:
        text = text.split('答案是')[1]
    return text


@ICL_EVALUATORS.register_module()
class OpsQAEvaluator(BaseEvaluator):

    def __init__(self):
        self.bleu = BleuEvaluator()
        self.rouge = RougeEvaluator()

    def score(self, predictions, references):
        bleu_score = self.bleu.score(predictions, references)
        rouge_score = self.rouge.score(predictions, references)
        bleu_score = {
            'bleu_' + key: value
            for key, value in bleu_score.items()
        }
        rouge_score = {
            'rouge_' + key: value
            for key, value in rouge_score.items()
        }
        bleu_score.update(rouge_score)
        return bleu_score
