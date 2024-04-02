import os
import json
import re

model_name = os.environ.get('MODEL')
model_path = os.environ.get('MODEL_PATH')
datasets = os.environ.get('DATASETS')
if model_name is None:
    raise NotImplementedError("Model name is none!")
if model_path is None:
    raise NotImplementedError("Model path is none!")
if datasets is None:
    raise NotImplementedError("Datasets is none!")
model_abbr = f"nm_{model_name.replace('-', '_')}"

# Check model_name in model_card
template_dict = {
    "qwen": """dict(
        round=[
            dict(role="HUMAN", begin='<|im_start|>user\\n', end='<|im_end|>'),
            dict(role="BOT", begin="<|im_start|>assistant\\n", end='<|im_end|>', generate=True),
        ],
        reserved_roles=[dict(role='SYSTEM', begin="<|im_start|>system\\nYou are a helpful assistant.", end="<|im_end|>"),]
    )""",
    "qwen1.5": """dict(
        round=[
            dict(role="HUMAN", begin='<|im_start|>user\\n', end='<|im_end|>'),
            dict(role="BOT", begin="<|im_start|>assistant\\n", end='<|im_end|>', generate=True),
        ],
        reserved_roles=[dict(role='SYSTEM', begin="<|im_start|>system\\nYou are a helpful assistant.", end="<|im_end|>"),]
    )""",
    "yi": """dict(
        round=[
            dict(role="HUMAN", begin='<|im_start|>user\\n', end='<|im_end|>'),
            dict(role="BOT", begin="<|im_start|>assistant\\n", end='<|im_end|>', generate=True),
        ],
    )""",
    "default": """dict(
        round=[
            dict(role="HUMAN", begin='<|im_start|>user\\n', end='<|im_end|>'),
            dict(role="BOT", begin="<|im_start|>assistant\\n", end='<|im_end|>', generate=True),
        ],
    )"""
}
meta_template = dict(
        round=[
            dict(role="HUMAN", begin='<|im_start|>user\\n', end='<|im_end|>'),
            dict(role="BOT", begin="<|im_start|>assistant\\n", end='<|im_end|>', generate=True),
        ],
    )

eos_tokens_dict = {
    "qwen": """[151643, 151645]""",
    "qwen1.5": """[151643, 151645]""",
    "yi": """[2, 7]""",
    "default": """[]"""
}

model_dict = {
    "huggingface": """dict(
            type=$INFERENCE_MODEL$,
            abbr=model_abbr,
            path=model_path,
            tokenizer_path=model_path,
            tokenizer_kwargs=dict(padding_side='left',
                                truncation_side='left',
                                trust_remote_code=True,
                                use_fast=False,),
            max_out_len=400,
            max_seq_len=2048,
            batch_size=8,
            model_kwargs=dict(device_map='auto', trust_remote_code=True),
            run_cfg=dict(num_gpus=$NUM_GPUS$, num_procs=1),
            meta_template=$META_TEMPLATE$,
            generation_kwargs=dict(eos_token_id=$EOS_TOKEN_IDS$)
        )""",
    "lmdeploy": """dict(
            type=$INFERENCE_MODEL$,
            abbr=model_abbr,
            path=model_path,
            engine_config=dict(session_len=2048,
                           max_batch_size=8),
            gen_config=dict(top_k=1, top_p=0.8,
                        max_new_tokens=100, stop_words=$EOS_TOKEN_IDS$),
            max_out_len=400,
            max_seq_len=2048,
            batch_size=8,
            run_cfg=dict(num_gpus=$NUM_GPUS$, num_procs=1),
            meta_template=$META_TEMPLATE$,
            end_str='<|im_end|>'
        )"""
}

def check_model_category(name: str, model_card):
    if 'qwen' in name.lower():
        if 'qwen1.5' in name.lower() or 'qwen-1.5' in name.lower():
            return 'qwen1.5'
        return 'qwen'
    if 'yi' in name.lower():
        return 'yi'
    if name in model_card and "CONFIG" in model_card[name] and "BASE_NAME" in model_card[name]["CONFIG"]:
        return check_model_category(model_card[name]["CONFIG"]["BASE_NAME"], model_card)
    return 'default'

def check_model_size(name: str, model_card):
    def extract_number(string):
        match = re.search(r'(\d+(?:\.\d+)?)[Bb](?=-|$)', string)
        if match:
            return match.group(1)
        else:
            return None
    model_size = extract_number(name)
    if model_size is not None:
        return float(model_size)
    if name in model_card and "CONFIG" in model_card[name] and "BASE_NAME" in model_card[name]["CONFIG"]:
        return check_model_size(model_card[name]["CONFIG"]["BASE_NAME"], model_card)
    return 14
    
def check_model_type(name: str, model_card):
    if 'sft' in name:
        return 'chat'
    if name in model_card and 'MODEL_TYPE' in model_card[name]:
        if model_card[name]['MODEL_TYPE'] == 'SFT':
            return 'chat'
        if model_card[name]['MODEL_TYPE'] == 'PT':
            return 'base'
        if model_card[name]['MODEL_TYPE'] == 'BASE':
            if 'chat' in name.lower():
                return 'chat'
    return 'base'

def check_meta_template():
    meta_template = "None"
    if not os.path.exists("/mnt/home/opsfm-xz/model_card.json"):
        print(f"[WARNING] model_card not found! Set meta_template to None!")
        return meta_template
    with open("/mnt/home/opsfm-xz/model_card.json") as f:
        model_card = json.load(f)
    model_category = check_model_category(model_name, model_card)
    model_type = check_model_type(model_name, model_card)
    print(f"[EVAL] Set model metatemplate type to {model_category}-{model_type}")
    if model_type == "chat":
        if model_category not in template_dict:
            model_category = 'default'
        meta_template = template_dict.get(model_category, "None")
    print(f"[EVAL] {meta_template=}")
    return meta_template

def determine_num_gpus():
    num_gpus = 1
    if not os.path.exists("/mnt/home/opsfm-xz/model_card.json"):
        print(f"[WARNING] model_card not found! Set num_gpus=1!")
        return num_gpus
    with open("/mnt/home/opsfm-xz/model_card.json") as f:
        model_card = json.load(f)
    model_size = check_model_size(model_name, model_card)
    if model_size > 35:
        num_gpus = 2
    print(f"[EVAL] {model_size=}, set {num_gpus=}")
    return str(num_gpus)

def determine_inference_model():
    # Check inference model
    inference_model = "HuggingFaceCausalLM"
    generation_kwargs = eos_tokens_dict['default']
    if not os.path.exists("/mnt/home/opsfm-xz/model_card.json"):
        print(f"[WARNING] model_card not found! Set inference_model to HuggingFaceCausalLM!")
    else:
        with open("/mnt/home/opsfm-xz/model_card.json") as f:
            model_card = json.load(f)
        model_category = check_model_category(model_name, model_card)
        if model_category in ['qwen', 'yi']:
            inference_model = 'TurboMindModel'
        # elif model_category in ['qwen1.5']:
        #     inference_model = 'LmdeployPytorchModel'
        
        # Check generation_kwargs
        if model_category in eos_tokens_dict:
            eos_tokens = eos_tokens_dict[model_category]
    print(f"[EVAL] Set {inference_model=}")
    print(f"[EVAL] Set {eos_tokens=}")
    
    # Check template type
    model_template_type = "lmdeploy"
    if inference_model in ['HuggingFaceCausalLM']:
        model_template_type = "huggingface"
    model_template = model_dict[model_template_type]
    print(f"[EVAL] Set {model_template_type=}")
    
    return inference_model, model_template, eos_tokens

inference_model, model_template, eos_tokens = determine_inference_model()

with open('/mnt/tenant-home_speed/lyh/opseval-configs/xz/runconfig_base.py') as fi, open('/mnt/tenant-home_speed/lyh/opseval-configs/xz/runconfig.py', 'wt') as fo:
    content = fi.read().replace('$MODEL_TEMPLATE$', model_template).replace('$MODEL$', model_name).replace('$MODEL_ABBR$', model_abbr).replace('$MODEL_PATH$', model_path).replace('$DATASETS$', datasets).replace('$META_TEMPLATE$', check_meta_template()).replace('$NUM_GPUS$', determine_num_gpus()).replace('$INFERENCE_MODEL$', inference_model).replace('$EOS_TOKEN_IDS$', eos_tokens)
    fo.write(content)
