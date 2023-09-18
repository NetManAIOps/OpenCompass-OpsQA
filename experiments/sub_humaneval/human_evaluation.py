import json, re, sys, os
from tqdm import tqdm
import pandas as pd
import random
from colorama import Fore, Back, Style

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

metrics = [
    "Fluency",
    "Accuracy",
    "Evidence"
]

def main(sub='en'):
    with open("/mnt/mfs/opsgpt/opencompass/experiments/sub_humaneval/merged.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    subset = data[sub]
    
    with open(f"/mnt/mfs/opsgpt/opencompass/experiments/sub_humaneval/{sub}.json","r") as f:
        score_dict = json.load(f)

    for qidx, q in subset.items():
        if str(qidx) in score_dict:
            continue

        print("============================================================")

        stats = dict()

        model_order =  random.sample(models, len(models))
        for model in model_order:
            # print(Fore.RED + 'Question: ' + Style.RESET_ALL + '\n'.join(q['origin_prompt'].split('\n')[1:-2]).strip())
            print(Fore.RED + 'Question: ' + Style.RESET_ALL + q['origin_prompt'].strip())
            print(Fore.RED + 'Key Points: ' + Style.RESET_ALL + Fore.GREEN + q['reference']['keypoint']+ Style.RESET_ALL)
            print(Fore.RED + 'Answer: ' + Style.RESET_ALL + Fore.GREEN + q['reference']['answer']+ Style.RESET_ALL)
            # pred = q[model].strip().split(' ')
            pred = q[model]
            if len(pred) > 500:
                print(pred[:500] + '...', Fore.RED + f'[TOTAL: {len(pred)}]' + Style.RESET_ALL)
                more = input("Do you wish to see the full prediction? y/[n]")
                if more == 'y':
                    print(pred)
            else:
                print(pred)
            
            scores = dict()

            for metric in metrics:
                score = -1
                while score not in [0, 1, 2, 3]:
                    try:
                        score = int(input(Fore.RED + f"{metric}: "+ Style.RESET_ALL ))
                    except:
                        score = -1
                    if score not in [0,1,2,3]:
                        print("Score must be in [0,3]!")
                scores[metric] = score
            print("------------------------------------------------------------")
            stats[model] = scores
        score_dict[qidx] = stats

        with open(f"/mnt/mfs/opsgpt/opencompass/experiments/sub_humaneval/{sub}.json","w") as f:
            json.dump(score_dict, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main(sub='zh')