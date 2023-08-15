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

single_ans_hint = 'You should select an appropriate option letter to answer this question. Example of a possible answer: A.'  # noqa
multiple_ans_hint = 'You should select all appropriate option letters separated by commas to answer this question. Example of a possible answer: B,C.'  # noqa

single_ans_hint_zh = '请选择正确答案的选项进行回答，例如：A、B、C、D'
multiple_ans_hint_zh = "请选出其中正确的选项，用空格或英文逗号分隔，例如：'A,B'、'A B C'"


@LOAD_DATASET.register_module()
class OReillyChoiceDataset(BaseDataset):

    @staticmethod
    def load(path: str, filename: str, sample_setting: dict = None):
        with open(osp.join(path, filename), encoding='utf-8') as f:
            json_data = json.load(f)

        sample_ids = None
        if sample_setting:
            if 'seed' in sample_setting:
                random.seed(sample_setting['seed'])
            else:
                random.seed(0)
            if 'load_list' in sample_setting:
                with open(sample_setting['load_list'], 'r',
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

        if sample_ids:
            json_data = [
                item for idx, item in enumerate(json_data) if idx in sample_ids
            ]

        raw_data = []
        for data in json_data:
            assert (data['type'] in [0, 1])
            choices_prompt = '\n'.join([
                f"{chr(ord('A')+idx)}: {choice}"
                for idx, choice in enumerate(data['choices'])
            ])
            item = {
                'id': data['id'],
                'qtype': 'single-answer multiple choice'
                if data['type'] == 0 else 'multiple-answer multiple choice',
                'question': data['question'],
                'topic':
                ','.join(data['topic']) if len(data['topic']) else 'Ops',
                'solution': data['solution'],
                'answer': data['answer'],  # data['answer'].replace(',', ''),
                'sol_len': len(data['solution']),
                'choices': choices_prompt,
                'hint':
                single_ans_hint if data['type'] == 0 else multiple_ans_hint
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
    s = text
    s += ' '
    matched = re.search(r'answer.+?[^\w]([a-zA-Z][^\w])', s.lower())
    if matched:
        s = s[matched.span()[0]:]
    if not re.match(r'[A-Z][^\w]', s):
        matched = re.search(r'[^\w][A-Z][^\w]', s)
        if matched:
            s = s[matched.span()[0] + 1:]

    s = s.strip() + ' '
    pattern = r'([A-Z][\s,]+(and)?[\s,]*)*[A-Z][^\w]'
    matched = re.match(pattern, s, re.S)
    prefix = matched.group() if matched else ''
    ans = list(
        set([
            c.upper() for c in re.split('[^A-Z]', prefix)
            if len(c) == 1 and c.isalpha()
        ]))
    ans.sort()
    return ','.join(ans)
