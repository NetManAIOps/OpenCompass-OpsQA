
import json, re, os, sys
import pandas as pd
from os import path as osp
import openai
from tqdm import tqdm
from configs.openai_key import peiqi_key
import openpyxl
import copy
import time

openai.api_base = "https://api.chatanywhere.com.cn/v1"
openai.api_key = peiqi_key

original_file = "/mnt/mfs/opsgpt/opencompass/experiments/oracle/oracle.xlsx"

def get_translation(problem, to_en=True):
    prompt = f'"{problem}"\n{"翻译成英文" if to_en else "翻译成中文"}'
    messages = [
        {'role': 'user', 'content': prompt}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=messages, 
        temperature=0)
    answer = response['choices'][0]['message']['content']
    return answer

categories = [
    '5G基础',
    '5G核心网',
    '5G无线接入网',
    '相关网络知识',
    '安全与规范'
]

loc = "/mnt/mfs/opsgpt/opencompass/experiments/zte/dataset"
target_loc = "/mnt/mfs/opsgpt/opencompass/experiments/zte/qa_only"

def main():
    for cat in categories:
        filename = os.path.join(loc, f"{cat}_zh.json")
        with open(filename, 'r') as f:
            data = json.load(f)
        en_data = []
        for q in tqdm(data):
            choices_prompt = '\n'.join([
                    f"{chr(ord('A')+idx)}: {choice}"
                    for idx, choice in enumerate(q['choices'])
                ])
            q['question'] = q['question'] + '\n' + choices_prompt

            while True:
                try:
                    translated = get_translation(q['question'], to_en=True)
                    break
                except:
                    continue
            en_q = copy.deepcopy(q)
            en_q['question'] = translated
            en_data.append(en_q)

            with open(os.path.join(target_loc, f"{cat}_zh.json"), 'w') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)

            with open(os.path.join(target_loc, f"{cat}_en.json"), 'w') as f:
                json.dump(en_data, f, indent=4, ensure_ascii=False)

            

    


if __name__ == "__main__":
    main()