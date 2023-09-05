import json, re, os, codecs
from os import path as osp
from tqdm import tqdm

def load_json(file):
    with open(file, 'r') as f:
        return json.load(f)

def escape_unicode(unicode_string):
    # unicode_string = unicode_string.replace("\n", "\\n")
    json_string = json.dumps(unicode_string)
    decoded_string = json.loads(json_string)
    # decoded_string = json.loads(f'"{unicode_string}"')
    return decoded_string


if __name__ == "__main__":
    translated = load_json("/mnt/mfs/opsgpt/evaluation/translated/translated_questions.json")
    for q in tqdm(translated):
        try:
            q['question'] = escape_unicode(q['question'])
            for choice in q['choices']:
                choice = escape_unicode(choice)
            q['solution'] = escape_unicode(q['solution'])
            for topic in q['topic']:
                topic = escape_unicode(topic)
        except:
            print(q)
            raise
    with open("/mnt/mfs/opsgpt/evaluation/translated/translated_escaped.json", 'w', encoding='utf-8') as f:
        json.dump(translated, f, ensure_ascii=False, indent=4)