
import json, re, os, sys
import pandas as pd
from os import path as osp
import openai
from tqdm import tqdm
from configs.openai_key import peiqi_key
import openpyxl
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

def main():
    # open xlsx from the variable 'original_file'
    # Load the workbook from the file path stored in the variable 'original_file'
    workbook = openpyxl.load_workbook(filename=original_file)
    
    df_zh = pd.read_excel(original_file, sheet_name='chinese')
    df_en = pd.read_excel(original_file, sheet_name="english")
    
    en_ds = []
    ze_ds = []
    zh_ds = []
    ez_ds = []

    for ridx, row in tqdm(df_zh.iterrows()):
        while True:
            try:
                translated = get_translation(row['题目'], to_en=True)
                break
            except:
                time.sleep(2)
        zh_ds.append({
            'question': row['题目'],
            'answer': ','.join(list(row['正确答案']))
        })
        ze_ds.append({
            'question': translated,
            'answer': ','.join(list(row['正确答案']))
        })
        with open("/mnt/mfs/opsgpt/opencompass/experiments/oracle/oracle_ze.json",'w') as f:
            json.dump(ze_ds, f, indent=4, ensure_ascii=False)
        with open("/mnt/mfs/opsgpt/opencompass/experiments/oracle/oracle_zh.json",'w') as f:
            json.dump(zh_ds, f, indent=4, ensure_ascii=False)

    
    for ridx, row in tqdm(df_en.iterrows()):
        while True:
            try:
                translated = get_translation(row['题目'], to_en=False)
                break
            except:
                time.sleep(2)
        en_ds.append({
            'question': row['题目'],
            'answer': ','.join(list(row['正确答案']))
        })
        ez_ds.append({
            'question': translated,
            'answer': ','.join(list(row['正确答案']))
        })
        with open("/mnt/mfs/opsgpt/opencompass/experiments/oracle/oracle_en.json",'w') as f:
            json.dump(en_ds, f, indent=4, ensure_ascii=False)
        with open("/mnt/mfs/opsgpt/opencompass/experiments/oracle/oracle_ez.json",'w') as f:
            json.dump(ez_ds, f, indent=4, ensure_ascii=False)
    


if __name__ == "__main__":
    main()