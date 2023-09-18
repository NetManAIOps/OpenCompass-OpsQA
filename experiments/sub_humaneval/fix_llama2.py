import os, sys, json, re

if __name__ == "__main__":
    with open("/mnt/mfs/opsgpt/opencompass/experiments/sub_humaneval/merged.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    for i in range(49, 23, -1):
        print(i)
        data['en'][str(i)]['llama-2-70b-int4'] = data['en'][str(i-1)]['llama-2-70b-int4']
        # data['zh'][str(i)]['llama-2-70b-int4'] = data['zh'][str(i-1)]['llama-2-70b-int4']
    data['en']['23']['llama-2-70b-int4'] = data['en']['21']['llama-2-70b-int4']
    # data['zh']['23']['llama-2-70b-int4'] = data['zh']['21']['llama-2-70b-int4']
    with open("/mnt/mfs/opsgpt/opencompass/experiments/sub_humaneval/merged2.json", 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)