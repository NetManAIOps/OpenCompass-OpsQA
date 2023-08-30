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


def translate_topic(topic):
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
        response = openai.ChatCompletion.create(model='gpt-3.5-turbo',
                                                messages=messages,
                                                temperature=0)
        answer = response['choices'][0]['message']['content']
        topic_dict[topic] = answer
        return answer


def translate_choice(choice):
    prompt = '你是一个英译汉的翻译机器人。你需要翻译一道选择题的某个选项，将下面这个选项翻译成中文，除了翻译之外不要有任何额外回复：\n'  # noqa
    messages = [{
        'role': 'system',
        'content': prompt
    }, {
        'role': 'user',
        'content': choice
    }]
    response = openai.ChatCompletion.create(model='gpt-3.5-turbo',
                                            messages=messages,
                                            temperature=0)
    answer = response['choices'][0]['message']['content']
    return answer


topic_dict = {}


def translate_topics(topics):
    return [translate_topic(topic) for topic in topics]


def translate_choices(choices):
    return [translate_choice(choice) for choice in choices]


def translate_sentence(origin):
    prompt = '你是一个英译汉的翻译机器人。将下面这个句子翻译成中文，其他不要有任何额外回复：\n'
    messages = [{
        'role': 'system',
        'content': prompt
    }, {
        'role': 'user',
        'content': origin
    }]
    response = openai.ChatCompletion.create(model='gpt-3.5-turbo',
                                            messages=messages,
                                            temperature=0)
    answer = response['choices'][0]['message']['content']
    return answer


def translate(q):
    # question
    question = translate_sentence(q['question'])
    # topic
    topics = translate_topics(q['topic'])
    # solution
    solution = translate_sentence(q['solution'])
    # choices
    choices = translate_choices(q['choices'])

    q.update({
        'question': question,
        'topic': topics,
        'solution': solution,
        'choices': choices
    })
    return q


def main():
    # 网络题目列表
    with open('/mnt/mfs/opsgpt/evaluation/ops-cert-eval/network_list.json',
              'r') as f:
        id_list = json.load(f)

    with open(
            '/mnt/mfs/opsgpt/opencompass/experiments/en_zh_comparison/translated_questions.json',  # noqa
            'r') as f:
        data = json.load(f)

    done_list = [q['id'] for q in data]

    wip_list = list(set(id_list) - set(done_list))
    print(f'{len(wip_list)} questions waiting to be translated...')

    # 获得题目
    questions = extract_questions(wip_list)

    for q in tqdm(questions):
        translated = translate(q)
        data.append(translated)
        if len(data) % 100 == 0:
            print(f'Saving... Current progress: {len(data)}')
            with open(
                    '/mnt/mfs/opsgpt/opencompass/experiments/en_zh_comparison/translated_questions.json',  # noqa
                    'w') as f:
                json.dump(data, f, indent=4)


if __name__ == '__main__':
    main()
