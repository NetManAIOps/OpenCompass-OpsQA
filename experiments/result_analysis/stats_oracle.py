from tools.utils import extract_answer, extract_questions
import sys, os, re, json
from tqdm import tqdm
import pandas as pd
import random
from collections import Counter

root = "/mnt/mfs/opsgpt/opencompass/outputs"

datasets = ['oracle']


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
    results = []
    for dataset in datasets:
        folder = os.path.join(root, dataset)
        folder = os.path.join(folder, os.listdir(folder)[0], "predictions")
        # Iterate through models
        for model in os.listdir(folder):
            for setting in os.listdir(os.path.join(folder, model)):
                if 'sc' not in setting:
                    continue
                filename = os.path.join(folder, model, setting)
                print(dataset, model, setting)
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                    tot = len(data)
                    sc_correct = 0
                    correct = 0

                    for q in data.values():
                        # check if q['thoughts'] exists
                        if 'thoughts' not in q:
                            q['thoughts'] = [[p] for p in q['prediction']]
                        
                        sc_correct += sc_cot(q)
                        correct += not_sc(q)

                    result = {
                        "model": model,
                        "dataset": setting.replace(".json", "").replace("network-", "").replace("oreilly-", ""),
                        "qtype": "single" if "single" in setting else "multiple",
                        "sc": "yes",
                        "acc": sc_correct / tot,
                        "chinese": "yes" if "zh" in setting else "no", 
                        "cot": "yes" if "cot" in setting else "no",
                        "shot": "Zero-shot" if "Zero-shot" in setting else "3-shot"
                    }
                    results.append(result)
                    result = {
                        "model": model,
                        "dataset": setting.replace(".json", "").replace("network-", "").replace("oreilly-", ""),
                        "qtype": "single" if "single" in setting else "multiple",
                        "acc": correct / tot,
                        "sc": "no",
                        "chinese": "yes" if "zh" in setting else "no", 
                        "cot": "yes" if "cot" in setting else "no",
                        "shot": "Zero-shot" if "Zero-shot" in setting else "3-shot"
                    }
                    results.append(result)
    df = pd.DataFrame(results)
    print(df)
    df.to_csv("oracle_results.csv")



if __name__ == "__main__":
    main()