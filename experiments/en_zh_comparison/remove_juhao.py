import json, re, sys, os

if __name__ == "__main__":
    names = [f'ch_{qtype}_{mode}.json' for qtype in ['single', 'multiple'] for mode in ['dev', 'test']]
    location = "/gpudata/home/cbh/opsgpt/evaluation/translated/v3"
    for name in names:
        path = os.path.join(location, name)
        print(path)
        with open(path, 'r') as f:
            data = json.load(f)
        for q in data:
            for idx, topic in enumerate(q['topic']):
                q['topic'][idx] = topic.replace('.','')
        with open(path, 'w') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
