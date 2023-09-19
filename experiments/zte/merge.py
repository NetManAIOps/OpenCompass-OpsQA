import json, re, os, sys

categories = [
    '5G基础',
    '5G核心网',
    '5G无线接入网',
    '相关网络知识',
    '安全与规范'
]

loc = "/mnt/mfs/opsgpt/opencompass/experiments/zte/qa_only"

if __name__ == "__main__":
    for lang in ['en', 'zh']:
        single = []
        multiple = []
        for cat in categories:
            filename = os.path.join(loc, f"{cat}_{lang}.json")
            with open(filename, 'r') as f:
                data = json.load(f)
            
            
            for q in data:
                q['question'] = q['question'].strip('"')
                if q['qtype'] == 0:
                    single.append(q)
                else:
                    multiple.append(q)
        with open(os.path.join(loc, f"zte_{lang}_single_test.json"), 'w') as f:
            json.dump(single, f, indent=4, ensure_ascii=False)
        with open(os.path.join(loc, f"zte_{lang}_multiple_test.json"), 'w') as f:
            json.dump(multiple, f, indent=4, ensure_ascii=False)
