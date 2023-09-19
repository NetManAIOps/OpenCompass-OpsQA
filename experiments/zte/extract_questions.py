import json, os, re
import pandas as pd
import openpyxl

file = "/mnt/mfs/opsgpt/opencompass/experiments/zte/zte.xlsx"

categories = [
    '5G基础',
    '5G核心网',
    '5G无线接入网',
    '相关网络知识',
    '安全与规范'
]

# categories = [
#     '5G Fundamentals',
#     '5G Core Network',
#     '5G Wireless Access Network',
#     'Related Network Knowledge',
#     'Security and Regulation'
# ]

dfs = dict()

def main():
    workbook = openpyxl.load_workbook(filename=file)
    for cat in categories:
        df = pd.read_excel(file, sheet_name=cat)
        dfs[cat] = df
    
    dss = dict()
    for cat, df in dfs.items():
        # print(df.columns) # Index(['题号', '题干', '选项1', '选项2', '选项3', '选项4', '选项5', '选项6', '答案', '题型'], dtype='object')
        data = []
        for ridx, row in df.iterrows():
            qid = f"{cat}-{row['题号']}"
            question = row['题干']

            choices = []
            for i in range(1, 7):
                col = f"选项{i}"
                if isinstance(row[col], str):
                    choices.append(row[col].strip())
                else:
                    break
            
            qtype = 0
            if row['题型'] == '多':
                qtype = 1

            answer = ','.join(list(row['答案'])).upper()
            if answer == 'T':
                answer = 'A'
            elif answer == 'F':
                answer = 'B'

            q = {
                'id': qid,
                'question': question,
                'choices': choices,
                'qtype': qtype,
                'answer': answer,
                'solution': ''
            }
            data.append(q)
        dss[cat] = data
        with open(f"/mnt/mfs/opsgpt/opencompass/experiments/zte/dataset/{cat}_zh.json", 'w') as f:
            json.dump(dss[cat], f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()