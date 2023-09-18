import json, os

loc = '/mnt/mfs/opsgpt/evaluation/oracle/'

if __name__ == "__main__":
    for lang in ['en', 'zh']:
        for split in ['dev', 'test']:
            filename = os.path.join(loc, f"oracle_{lang}_{split}.json")
            with open(filename, 'r') as f:
                data = json.load(f)
            
            for q in data:
                q['solution'] = ''
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)