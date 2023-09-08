import concurrent.futures  # noqa
import json  # noqa
import os  # noqa
import re  # noqa
import sys  # noqa
from os import path as osp  # noqa

import openai
import pandas as pd  # noqa
from tqdm import tqdm

from configs.openai_key import peiqi_key
from tools.utils import extract_questions

openai.api_base = 'https://api.chatanywhere.cn/v1'
openai.api_key = peiqi_key


def ask_step_by_step(topic):
    if topic in topic_dict:
        return topic_dict[topic]
    else:
        prompt = '你是一个英译汉的翻译机器人。将下面这个单词或短语翻译成中文，其他不要有任何额外回复：\n'
        messages = [{
            'role': 'system',
            'content': prompt
        }, {
            'role': 'user',
            'content': topic
        }]
        response = openai.ChatCompletion.create(model='gpt-4',
                                                messages=messages,
                                                temperature=0)
        answer = response['choices'][0]['message']['content']
        topic_dict[topic] = answer
        return answer


def main():
    dev_list = ['multiple_dev.json', 'single_dev.json']
    folder = '/mnt/mfs/opsgpt/evaluation/ops-cert-eval/v3'
    for devset in dev_list:
        filename = osp.join(folder, devset)
        with open(filename, 'r') as f:
            data = json.load(f)

        cot_data = []
        cot_prompt = []

        for q in tqdm(data):
            qtype = 'single-answer multiple choice' if q[
                'type'] == 0 else 'multiple-answer multiple choice'
            choices = '\n'.join([
                f"{chr(ord('A')+idx)}: {choice}"
                for idx, choice in enumerate(q['choices'])
            ])
            topic = ','.join(q['topic']) if len(q['topic']) else 'Ops'
            # prompt = f"Here is a {qtype} question about {topic}.\n{q['question']}\n{choices}\nLet's think step by step.\n"
            prompt = f"I will give you a multi-choice question. You need to analyze each choice of the question and then give an answer.\nQuestion: {q['question']}\n{choices}\n"
            messages = [{'role': 'user', 'content': prompt}]
            response = openai.ChatCompletion.create(model='gpt-4',
                                                    messages=messages,
                                                    temperature=0)
            answer = response['choices'][0]['message']['content']
            q['original_solution'] = q['solution']
            q['solution'] = answer
            cot_data.append(q)
            cot_prompt.append(prompt)

        with open(filename.replace('dev.json', 'cot_dev.json'), 'w') as f:
            json.dump(cot_data, f, indent=4)

        with open(filename.replace('dev.json', 'cot_dev_prompt.json'),
                  'w') as f:
            json.dump(cot_prompt, f, indent=4)


if __name__ == '__main__':
    main()
