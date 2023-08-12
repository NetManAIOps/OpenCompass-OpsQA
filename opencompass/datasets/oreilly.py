import csv  # noqa
import json
import os.path as osp  # noqa

from datasets import Dataset, DatasetDict  # noqa

from opencompass.openicl.icl_evaluator import (BaseEvaluator, BleuEvaluator,
                                               RougeEvaluator)
from opencompass.registry import TEXT_POSTPROCESSORS  # noqa
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class OReillyChoiceDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        with open(path) as f:
            json_data = json.load(f)
        raw_data = []
        for data in json_data:
            assert (data['type'] in [0, 1])
            choices_prompt = '\n'.join([
                f"{chr(ord('A')+idx)}: {choice}"
                for idx, choice in enumerate(data['choices'])
            ])
            item = {
                'question': data['question'],
                'topic':
                ','.join(data['topic']) if len(data['topic']) else 'Ops',
                'solution': data['solution'],
                'answer': data['answer'],  # data['answer'].replace(',', ''),
                'sol_len': len(data['solution']),
                'choices': choices_prompt
            }
            raw_data.append(item)
        return Dataset.from_list(raw_data)


@ICL_EVALUATORS.register_module()
class OReillyEvaluator(BaseEvaluator):

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


@TEXT_POSTPROCESSORS.register_module('oreilly-choice')
def oreilly_choice_postprocess(text: str) -> str:
    ans = []
    s = text.strip()
    while s and s[0].isalpha():
        nxt = s[1:].strip()
        if not nxt or not nxt[0].isalnum():
            ans.append(s[0])
        if not nxt or nxt[0] != ',' or len(nxt) <= 1:
            break
        s = nxt[1:].strip()
    ans.sort()
    return ','.join(ans)
