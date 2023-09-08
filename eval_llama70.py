import asyncio
import json
import sys
import os
import time
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


try:
    import websockets
except ImportError:
    print("Websockets package not found. Make sure it's installed.")

HOST = 'gpu5:21698'
URI = f'ws://{HOST}/api/v1/chat-stream'

async def run(user_input, history):
    # Note: the selected defaults change from time to time.
    request = {
        'user_input': user_input,
        'history': history,
        
        'max_new_tokens': 1024,
        # 'auto_max_new_tokens': True,

        'mode': 'instruct',  # Valid options: 'chat', 'chat-instruct', 'instruct'
        'character': None,
        # 'instruction_template': 'Llama-v2',  # Will get autodetected if unset
        'instruction_template': 'Galactica',  # Will get autodetected if unset
        # 'instruction_template': 'Chinese',  # Will get autodetected if unset
        'your_name': 'user',
        'name1': 'user',
        'name2': 'assistant',
        # 'context': '',
        # 'context': 'This is a conversation with your Assistant. It is a computer program designed to help you with various tasks such as answering questions, providing recommendations, and helping with decision making. You can ask it anything you want and it will do its best to give you accurate and relevant information.',
        'regenerate': False,
        '_continue': False,
        # 'chat_instruct_command': 'Continue the chat dialogue below. Write a single reply for the character "<|character|>".\n\n<|prompt|>',
        'chat_instruct_command': '<|name1|>:<|prompt|><|name2|>:',

        'do_sample': True,
        'temperature': 0.7,
        'top_p': 0.9,
        'typical_p': 1,
        'epsilon_cutoff': 0,  # In units of 1e-4
        'eta_cutoff': 0,  # In units of 1e-4
        'tfs': 1,
        'top_a': 0,
        'repetition_penalty': 1.1,
        'repetition_penalty_range': 0,
        'top_k': 5,
        'min_length': 0,
        'no_repeat_ngram_size': 0,
        'num_beams': 1,
        'penalty_alpha': 0,
        'length_penalty': 1,
        'early_stopping': False,
        'mirostat_mode': 0,
        'mirostat_tau': 5,
        'mirostat_eta': 0.1,
        'guidance_scale': 1,
        'negative_prompt': '',

        'seed': -1,
        'add_bos_token': True,
        'truncation_length': 4096,
        'ban_eos_token': False,
        'skip_special_tokens': True,
        'stopping_strings': [],
    }

    async with websockets.connect(URI, ping_interval=None) as websocket:
        await websocket.send(json.dumps(request))

        while True:
            incoming_data = await websocket.recv()
            incoming_data = json.loads(incoming_data)

            if incoming_data['event'] == 'text_stream':
                yield incoming_data['history']
            elif incoming_data['event'] == 'stream_end':
                return


async def generate_response(user_input, history, output=False):
    async for new_history in run(user_input, history):
        cur_message = new_history['visible'][-1][1]
        if output:
            os.system('clear' if os.name == 'posix' else 'cls')
            print(cur_message, end='')
            sys.stdout.flush()
    return cur_message


# single_ans_example = 'You are configuring the mail app on an iPhone to use an `[Outlook.com](http://outlook.com)` email address. What configuration information do you need to enter to establish connectivity?\nA: Email address and password\nB: Email address, password, and server name\nC: Email address, password, server name, and mail protocol\nD: Email address, password, server name or IP address, and mail protocol\n\nAnswer: A\n\nQuestion: Which type of smartphone display has fewer layers, providing more flexibility, a better viewing angle, and excellent color as compared to other technologies?\nA: IPS\nB: OLED\nC: LED\nD: TN\n\nAnswer: B\n\nQuestion: You are purchasing a mobile device that allows you to use multi-finger gestures to interact with the device. What type of touchscreen technology does this device most likely use?\nA: Capacitive\nB: Infrared\nC: SAW\nD: Resistive\n\nAnswer: A\n\n'
multiple_ans_example = "A coworker is having a problem with their laptop and has asked you to fix it. When an external keyboard is plugged in, the laptop works just fine, but without it strange characters appear on the screen when typing. Which of the following are likely causes? (Choose two.)\nA: The driver is corrupted and needs to be updated/replaced.\nB: The ribbon cable is partially disconnected and needs to be reseated.\nC: The laptop needs to be replaced.\nD: There is debris under the keys.\n\nAnswer: A,B\n\nQuestion: Your boss has asked you to make sure that each of the company's laptops has a biometric scanner as an extra measure of security. Which of the following are biometric devices that are commonly found and can be configured on laptops? (Choose two.)\nA: ID card scanner\nB: Face ID\nC: Fingerprint reader\nD: Retina scanner\n\nAnswer: B,C\n\nQuestion: Your customer has an iPhone 8 and wants to read and write NFC tags with it. What advice will you give them? (Choose two.)\nA: Their iPhone will only work with Apple Pay.\nB: They need to upgrade to iPhone X to write NFC tags.\nC: With iOS 13 or better on their device, they'll be able to read and write NFC tags using a third-party app.\nD: iPhone 12 and iPhone 13 can read NFC tags simply by holding the phone over the tag.\n\nAnswer: C,D\n\n"
single_ans_example = '哪个Windows实用程序允许您查看由系统事件、应用程序或登录失败生成的错误消息？\nA: 系统信息\nB: Windows Defender\n C: 事件查看器\nD: 磁盘管理\n\nAnswer: C\n\nQuestion: 您正在解决网络连接问题，并想查看本地计算机是否能够访问网络上的特定服务器。您将使用什么命令？A: `ping`\nB: `tracert`\n C: `ipconfig`\nD: `pathping`\n\nAnswer: A\n\nQuestion: 您在一台32位处理器的台式电脑上安装了Windows 10 32位操作系统。在安装更多内存后，总共达到12GB，但操作系统只能识别4GB以下的内存。问题出在哪里？A: 您需要告诉BIOS额外的内存已经存在。\nB: 32位操作系统最多只支持4GB内存。\n C: 您需要进入设备管理器添加内存。\nD: 您需要更换主板。\n\nAnswer: B\n\n'


def submit_oreilly(subset, index):
    output_dir = '/home/v-xll22/OpsGPT/OpenCompass-OpsQA/outputs/few_shot/llama2-70b-int4/chinese'
    os.makedirs(output_dir, exist_ok=True)
    i = 0
    few_shot = True
    output = {}
    history = {'internal': [], 'visible': []}
    for data in tqdm(subset):
        choices_prompt = '\n'.join([
            f"{chr(ord('A')+idx)}: {choice}"
            for idx, choice in enumerate(data['choices'])
        ])
        if few_shot:
            if data['type'] == 0:
                example_str = single_ans_example
            else:
                example_str = multiple_ans_example
        else:
            example_str = ''
        item = {
            'id': data['id'],
            'qtype': 'single-answer multiple choice'
            if data['type'] == 0 else 'multiple-answer multiple choice',
            'question': data['question'],
            'topic': ','.join(data['topic']) if len(data['topic']) else '运维',
            'answer': data['answer'],
            'choices': choices_prompt,
            'examples': example_str
        }
        # prompt = f"{item['examples']}Answer the following {item['qtype']} question about {item['topic']}.\nQuestion: {item['question']}\n{item['choices']}"
        prompt = f"{item['examples']}以下是关于{item['topic']}的单项选择题，请选出其中的正确答案。\nQuesion: {item['question']}\n{item['choices']}"
        res = asyncio.run(generate_response(prompt, history, output=False))
        # prediction = oreilly_choice_postprocess(res)
        output[str(i)] = {
            "origin_prompt": prompt,
            "prediction": res,
            "reference": {
                "answer": item['answer'],
                "id": item['id']
            }
        }
        i += 1
    with open(os.path.join(output_dir, f'output_{index}.json'), 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=4, ensure_ascii=False)


def eval_oreilly():
    # dataset_path = '/mnt/mfs/opsgpt/evaluation/ops-cert-eval/fewshot.json'
    dataset_path = '/mnt/mfs/opsgpt/evaluation/translated/chinese_test.json'
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    max_num_workers = 16
    subsets = []
    for i in range(max_num_workers):
        begin = len(dataset) // max_num_workers * i
        if i != max_num_workers - 1:
            end = len(dataset) // max_num_workers * (i+1)
        else:
            end = len(dataset)
        subsets.append(dataset[begin: end])
    # for i, subset in enumerate(subsets):
    #     asyncio.run(submit(subset, i))
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=max_num_workers) as executor:
        executor.map(submit_oreilly, subsets, range(len(subsets)))
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("执行时间：", elapsed_time, "秒")


def submit_sub(subset, index):
    output_dir = '/home/v-xll22/OpsGPT/OpenCompass-OpsQA/outputs_sub/llama2-70b-int4'
    os.makedirs(output_dir, exist_ok=True)
    i = 0
    output = {}
    history = {'internal': [], 'visible': []}
    for data in tqdm(subset):
        item = {
            'question': data['question'],
            'topic': data['topic'],
            'task': data['task'],
            'answer': data['answer']
        }
        prompt = f"回答下面关于{item['topic']}的{item['task']}问题。\n{item['question']}\n答:\n"
        res = asyncio.run(generate_response(prompt, history, output=False))
        output[str(i)] = {
            "origin_prompt": prompt,
            "prediction": res,
            "reference": {
                "answer": item['answer']
            }
        }
        i += 1
    with open(os.path.join(output_dir, f'output_{index}.json'), 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=4, ensure_ascii=False)


def eval_sub():
    # dataset_path = '/mnt/mfs/opsgpt/evaluation/ops-cert-eval/fewshot.json'
    dataset_path = '/home/v-xll22/OpsGPT/opsqa.json'
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    max_num_workers = 5
    subsets = []
    for i in range(max_num_workers):
        begin = len(dataset) // max_num_workers * i
        if i != max_num_workers - 1:
            end = len(dataset) // max_num_workers * (i+1)
        else:
            end = len(dataset)
        subsets.append(dataset[begin: end])
    # for i, subset in enumerate(subsets):
    #     asyncio.run(submit(subset, i))
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=max_num_workers) as executor:
        executor.map(submit_sub, subsets, range(len(subsets)))
    # for i in range(len(subsets)):
    #     submit_sub(subsets[i], i)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("执行时间：", elapsed_time, "秒")


if __name__ == '__main__':
    # eval_oreilly()
    eval_sub()