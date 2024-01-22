import os.path as osp  # noqa
import json
from datasets import Dataset # noqa

from opencompass.openicl.icl_evaluator import (BaseEvaluator, BleuEvaluator,
                                               RougeEvaluator)
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from .base import BaseDataset


@LOAD_DATASET.register_module()
class OpsQADataset(BaseDataset):

    @staticmethod
    def load(path: str):
        with open(osp.join(path), encoding='utf-8') as f:
            json_data = json.load(f)
        raw_data = []
        for data in json_data:
            item = {
                'question': data['question'],
                'topic': data['topic'],
                'task': data['task'],
                'keypoint': data['keypoint'],
                'answer': data['answer']
            }
            raw_data.append(item)
        return Dataset.from_list(raw_data)

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