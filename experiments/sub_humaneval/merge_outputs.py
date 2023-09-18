import json, re, sys, os
from tqdm import tqdm
import pandas as pd


loc = "/mnt/mfs/opsgpt/opencompass/archive/v-xll22/outputs_sub/"

subs = ['zh', 'en']

file_list = [f'OpsQA_{idx}.json' for idx in range(4)]

models = [
    "baichuan-13b-chat",
    "chinese-alpaca-2-13b",
    "GPT-3.5-turbo",     
    "llama-2-13b-chat",  
    "qwen_chat_7b",
    "chatglm2-6b",        
    "chinese-llama-2-13b",   
    "internlm_chat_7b",  
    "llama-2-70b-int4"
]

merged = {
    "en": {str(i):dict() for i in range(50)},
    "zh": {str(i):dict() for i in range(50)},
}

def merge_outputs():
    for sub in subs:
        for fidx, file in enumerate(file_list):
            for model in models:
                filename = os.path.join(loc, sub, model, file)
                try:
                    with open(filename, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except FileNotFoundError:
                    print("No ", filename)
                    continue
                for qidx, q in data.items():
                    qidx = int(qidx)+fidx*13
                    q[model] = q['prediction']
                    q.pop('prediction')
                    merged[sub][str(qidx)].update(q)
    return merged


if __name__ == "__main__":
    result = merge_outputs()
    with open("/mnt/mfs/opsgpt/opencompass/experiments/sub_humaneval/merged.json", 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)