import json, os

loc = '/mnt/mfs/opsgpt/evaluation/oracle/'

if __name__ == "__main__":
    for lang in ['en', 'zh']:
        for split in ['dev', 'test']:
            single = []
            multiple = []

            filename = os.path.join(loc, f"oracle_{lang}_{split}.json")
            with open(filename, 'r') as f:
                data = json.load(f)
            
            for q in data:
                if len(q['answer'].split(',')) > 1:
                    multiple.append(q)
                else:
                    single.append(q)
            
            with open(os.path.join(loc, f"oracle_{lang}_single_{split}.json"), 'w') as f:
                json.dump(single, f, indent=4, ensure_ascii=False)
            with open(os.path.join(loc, f"oracle_{lang}_multiple_{split}.json"), 'w') as f:
                json.dump(multiple, f, indent=4, ensure_ascii=False)