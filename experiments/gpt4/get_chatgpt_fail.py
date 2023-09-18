from tools.utils import extract_answer, extract_questions
import sys, os, re, json
from tqdm import tqdm
import pandas as pd
import random
from collections import Counter

root = "/mnt/mfs/opsgpt/opencompass/outputs"

datasets = ["network_gpt"]

files = ['network-zh-3shot-sc+cot-multiple.json', 'network-zh-3shot-sc+cot-single.json', 
         'oreilly-3shot-sc+cot-multiple.json', 'oreilly-3shot-sc+cot-single.json']


def extract_answer(text: str) -> str:
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
    # matched = re.match(pattern, s, re.S)
    matched = list(re.finditer(pattern, s, re.S))
    first = matched[0].group() if matched else ''
    last = matched[-1].group() if matched else ''

    def process_prefix(prefix):
        ans = list(
            set([
                c.upper() for c in re.split('[^A-Z]', prefix)
                if len(c) == 1 and c.isalpha()
            ]))
        ans.sort()
        return ','.join(ans)
    
    return [process_prefix(a) for a in [first, last]]

def sc_cot(q):
    answer = q['reference']['answer']
    sc_thoughts = q['thoughts']
    assert isinstance(sc_thoughts, list)

    right_count = 0

    sc_choices = []

    # Iterate through sc cases
    for sc_case in sc_thoughts:
        assert isinstance(sc_case, list)
        # Iterate through stage of thoughts:
        thought_choices = []
        for thought in sc_case:
            assert isinstance(thought, str)
            prediction = extract_answer(thought)
            thought_choices += prediction
            if any([d == answer for d in prediction]):    # 所有thought中只要有一个正确即可
                right_count += 1
                thought_choices = [answer]
                break
        # print(Counter(thought_choices).most_common(1)[0][0])
        sc_choices.append(Counter(thought_choices).most_common(1)[0][0])
    try:
        # print(Counter(sc_choices).most_common(1)[0], answer)
        if Counter(sc_choices).most_common(1)[0][0] == answer:
            return 1
    except:
        print(sc_choices)
        print(q)
        raise
    return 0

def not_sc(q):
    answer = q['reference']['answer']
    sc_thoughts = q['thoughts']
    assert isinstance(sc_thoughts, list)

    right_count = 0

    # Iterate through sc cases
    sc_case = random.choice(sc_thoughts)
    # for sc_case in sc_thoughts[:1]:
    assert isinstance(sc_case, list)
    # Iterate through stage of thoughts:
    for thought in sc_case:
        assert isinstance(thought, str)
        prediction = extract_answer(thought)
        if any([d == answer for d in prediction]):    # 所有thought中只要有一个正确即可
            right_count += 1
            break
    if right_count > 0:
        return 1
    return 0

def main():
    

    for dataset in datasets:
        folder = os.path.join(root, dataset)
        folder = os.path.join(folder, os.listdir(folder)[0], "predictions")
        # Iterate through models
        for model in os.listdir(folder):
            # for setting in os.listdir(os.path.join(folder, model)):
            for setting in files:
                failcase = []
                    
                single_cnt = 0
                if 'sc' not in setting:
                    continue
                filename = os.path.join(folder, model, setting)
                print(dataset, model, setting)
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                    for q in data.values():
                        # check if q['thoughts'] exists
                        if 'thoughts' not in q:
                            q['thoughts'] = [[p] for p in q['prediction']]
                        
                        if not sc_cot(q):
                            failcase.append(q['reference']['id'])
                            single_cnt += 1
                print("this file: ", single_cnt)

                with open(f'/mnt/mfs/opsgpt/opencompass/experiments/gpt4/chatgpt_fail_{setting}', 'w') as f:
                    json.dump(failcase, f, indent=4, ensure_ascii=False)
    # print(failcase)
    print(len(failcase))


"""
network_gpt GPT-3.5-turbo network-zh-3shot-sc+cot-multiple.json
this file:  440
network_gpt GPT-3.5-turbo network-zh-3shot-sc+cot-single.json
this file:  831
network_gpt GPT-3.5-turbo oreilly-3shot-sc+cot-multiple.json
this file:  380
network_gpt GPT-3.5-turbo oreilly-3shot-sc+cot-single.json
this file:  693
693
"""

if __name__ == "__main__":
    main()