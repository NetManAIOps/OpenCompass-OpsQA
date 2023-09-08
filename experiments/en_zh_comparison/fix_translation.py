import json, re, os, codecs
from os import path as osp
from tools.utils import extract_questions
from tqdm import tqdm

reduntant_indicator = [
    '翻译是',
    '翻译为'
]

fail_indicator = [
    '无法翻译',
    '抱歉'
]

redundant_punct = [
    '：',
    '”',
    '“'
]

import re
import ipaddress

def fix_special_cases(choices, original_choices):
    device_org = ['Hub', 'Switch', 'Cable', 'Router', 
                  'NIC', 'Bridge', 'Modem', 'Repeater',
                  'Access Point', 'Server', 'Wireless access point',
                  'Hubs', 'Switches', 'Cables', 'Routers', 
                  'NICs', 'Bridges', 'Modems', 'Repeaters',
                  'Servers', 'Wireless access points',
                  'load balancer'
                  ]
    device_tra = ['集线器', '交换机', '电缆', '路由器', 
                  '网络接口卡', '网桥', '调制调解器', '中继器',
                  '接入点', '服务器', '无线接入点',
                  '集线器', '交换机', '电缆', '路由器', 
                  '网络接口卡', '网桥', '调制调解器', '中继器',
                  '服务器', '无线接入点',
                  '负载均衡器'
                  ]
    device_dict = {org.lower():tra for org, tra in zip(device_org, device_tra)}
    layer_org = ['Physical', 'Data link', 'Network', 'Transport', 'Session', 'Presentation', 'Application']
    layer_tra = ['物理层', '数据链路层', '网络层', '传输层', '会话层', '表示层', '应用层']
    layer_dict = {org.lower():tra for org, tra in zip(layer_org, layer_tra)}

    all_can_fixed = all([(ch.lower() in device_dict or ch.lower() in layer_dict) for ch in original_choices])
    if all_can_fixed:
        translated = []
        for ch in original_choices:
            if ch.lower() in device_dict:
                translated.append(device_dict[ch.lower()])
            else:
                translated.append(layer_dict[ch.lower()])
        return translated
    else:
        return None


def check_string_type(input_string):
    # 检查是否只包含大写英文字母
    if re.match("^[A-Z]+$", input_string):
        return True
    
    # 检查是否不包含小写英文字母
    if not re.search(r'[a-z]', input_string):
        return True
    
    # 检查是否包含`
    if '`' in input_string:
        return True
    
    if re.match(r"^(https?|ftp)://[^\s/$.?#].[^\s]*$", input_string):
        return True
    
    try:
        # 尝试解析输入字符串为IPv4或IPv6地址
        ip = ipaddress.ip_address(input_string)
        if ip.version == 4:
            return True
        elif ip.version == 6:
            return True
    except ValueError:
        pass
    
    return False

def check_choices_need_translation(choices):
    """see if the choices need to be translated or not
    """
    cnt = 0
    for choice in choices:
        if check_string_type(choice):
            cnt += 1
    if cnt >= (len(choices) / 2):
        return False
    return True

def check_choice_failed(choices, original_choices, qid):
    for choice in choices:
        if '无法' in choice or '抱歉':
            print(f"id: {qid} ORIGINAL ", original_choices)
            # drop_or_not = input()
            # if drop_or_not:
            #     return True
            # else:
            #     return False
            return False


# # 测试函数
# input_str1 = "ABCXYZ"
# input_str2 = "123.456.789.0"
# input_str3 = "2001:0db8:85a3:0000:0000:8a2e:0370:7334"

# print(check_string_type(input_str1))  # 输出: 只包含大写英文字母
# print(check_string_type(input_str2))  # 输出: 未知类型
# print(check_string_type(input_str3))  # 输出: IPv6地址



def load_json(file):
    with open(file, 'r') as f:
        return json.load(f)
    
count = 0

def detect_redundant(string):
    if '翻译' in string or '抱歉' in string:
        # print(string)
        return True
    return False

if __name__ == "__main__":
    # 网络题目列表
    with open('/mnt/mfs/opsgpt/evaluation/ops-cert-eval/network_list.json',
              'r') as f:
        id_list = json.load(f)

    failed = []
    failed_org = []
    original = extract_questions(id_list)
    original_dict = {q['id']: q for q in original}
    translated = load_json("/mnt/mfs/opsgpt/evaluation/translated/translated_questions_v2.json")
    for q in tqdm(translated):
        flag = False
        try:
            flag |= detect_redundant(q['question'])
            for choice in q['choices']:
                flag |= detect_redundant(choice)
            flag |= detect_redundant(q['solution'])
            for topic in q['topic']:
                flag |= detect_redundant(topic)
        except:
            print(q)
            raise
        if flag:
            q_org = original_dict[q['id']]
            if not check_choices_need_translation(q_org['choices']):
                q['choices'] = q_org['choices']

            if check_choice_failed(q['choices'], q_org['choices'], q['id']):
                q['choices'] = q_org['choices']
                pass

            special_case_fix = fix_special_cases(q['choices'], q_org['choices'])
            if special_case_fix:
                q['choices'] = special_case_fix

            failed.append(q)
            failed_org.append(q_org)
            # print(q)
            # print("====================")
            count += 1

    with open("/mnt/mfs/opsgpt/evaluation/translated/translated_questions_v2.json", 'w') as f:
        json.dump(translated, f, ensure_ascii=False, indent=4)
    with open("/mnt/mfs/opsgpt/opencompass/experiments/en_zh_comparison/failed.json", 'w') as f:
        json.dump(failed, f, ensure_ascii=False, indent=4)
    with open("/mnt/mfs/opsgpt/opencompass/experiments/en_zh_comparison/failed_org.json", 'w') as f:
        json.dump(failed_org, f, ensure_ascii=False, indent=4)    
    print(count)