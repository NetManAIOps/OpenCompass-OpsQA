from opencompass.models import HuggingFace, HuggingFaceCausalLM
from mmengine.config import read_base
with read_base():
    from ...paths import ROOT_DIR
    from ..model_template import get_default_model, get_vllm_model
    
qwen_meta_template = dict(
    round=[
        dict(role="HUMAN", begin='<|im_start|>user\n', end='<|im_end|>'),
        dict(role="BOT", begin="<|im_start|>assistant\n", end='<|im_end|>', generate=True),
    ],
)

qwen1_5_0_5b_base = get_default_model(abbr="qwen1.5-0.5b-base", path=f"{ROOT_DIR}models/Qwen/Qwen1.5-0.5B")
qwen1_5_0_5b_chat = get_default_model(abbr="qwen1.5-0.5b-chat", path=f"{ROOT_DIR}models/Qwen/Qwen1.5-0.5B-Chat")
qwen1_5_1_8b_base = get_default_model(abbr="qwen1.5-1.8b-base", path=f"{ROOT_DIR}models/Qwen/Qwen1.5-1.8B")
qwen1_5_1_8b_chat = get_default_model(abbr="qwen1.5-1.8b-chat", path=f"{ROOT_DIR}models/Qwen/Qwen1.5-1.8B-Chat")
qwen1_5_4b_base = get_default_model(abbr="qwen1.5-4b-base", path=f"{ROOT_DIR}models/Qwen/Qwen1.5-4B")
qwen1_5_4b_chat = get_default_model(abbr="qwen1.5-4b-chat", path=f"{ROOT_DIR}models/Qwen/Qwen1.5-4B-Chat")

qwen1_5_base_models = [
    get_default_model(abbr=f"qwen1.5-{quant}b-base", path=f"{ROOT_DIR}models/Qwen/Qwen1.5-{quant}B", num_gpus=1 if "72" not in quant else 2)
        for quant in ["0.5", "1.8", "4", "7", "14", "72"]
]

qwen1_5_chat_models = [
    get_default_model(abbr=f"qwen1.5-{quant}b-chat", 
                      meta_template=qwen_meta_template, 
                      num_gpus=1 if "72" not in quant else 2,
                      path=f"{ROOT_DIR}models/Qwen/Qwen1.5-{quant}B-Chat")
        for quant in ["0.5", "1.8", "4", "7", "14", "72"]
]

qwen_1_8b_base = dict(
        type=HuggingFaceCausalLM,
        abbr='qwen-1.8b-base',
        path="/mnt/tenant-home_speed/gaozhengwei/projects/LLM/models/Qwen/Qwen-1_8B",
        tokenizer_path='/mnt/tenant-home_speed/gaozhengwei/projects/LLM/models/Qwen/Qwen-1_8B',
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              trust_remote_code=True,
                              use_fast=False,),
        max_out_len=400,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )

qwen_1_8b_chat = dict(
        type=HuggingFaceCausalLM,
        abbr='qwen-1.8b-chat',
        path="/mnt/tenant-home_speed/gaozhengwei/projects/LLM/models/Qwen/Qwen-1_8B-Chat",
        tokenizer_path='/mnt/tenant-home_speed/gaozhengwei/projects/LLM/models/Qwen/Qwen-1_8B-Chat',
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              trust_remote_code=True,
                              use_fast=False,),
        max_out_len=400,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )


qwen_14b_base = dict(
        type=HuggingFaceCausalLM,
        abbr='qwen-14b-base',
        path="/mnt/tenant-home_speed/gaozhengwei/projects/LLM/models/Qwen/Qwen-14B",
        tokenizer_path='/mnt/tenant-home_speed/gaozhengwei/projects/LLM/models/Qwen/Qwen-14B',
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              trust_remote_code=True,
                              use_fast=False,),
        max_out_len=400,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )

qwen_14b_chat = dict(
        type=HuggingFaceCausalLM,
        abbr='qwen-14b-chat',
        path="/mnt/tenant-home_speed/gaozhengwei/projects/LLM/models/Qwen/Qwen-14B-Chat",
        tokenizer_path='/mnt/tenant-home_speed/gaozhengwei/projects/LLM/models/Qwen/Qwen-14B-Chat',
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              trust_remote_code=True,
                              use_fast=False,),
        max_out_len=400,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )

qwen1_5_14b_base = dict(
        type=HuggingFaceCausalLM,
        abbr='qwen1.5-14b-base',
        path=f"{ROOT_DIR}models/qwen/Qwen1.5-14B",
        tokenizer_path=f"{ROOT_DIR}models/Qwen/Qwen1.5-14B",
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              trust_remote_code=True,
                              use_fast=False,),
        max_out_len=400,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )

qwen1_5_14b_chat = dict(
        type=HuggingFaceCausalLM,
        abbr='qwen1.5-14b-chat',
        path=f"{ROOT_DIR}models/qwen/Qwen1.5-14B-Chat",
        tokenizer_path=f"{ROOT_DIR}models/Qwen/Qwen1.5-14B-Chat",
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              trust_remote_code=True,
                              use_fast=False,),
        max_out_len=400,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )

qwen1_5_base_vllm_models = [
    get_vllm_model(
        abbr=f"qwen1.5-{quant}b-base", 
        path=f"{ROOT_DIR}models/Qwen/Qwen1.5-{quant}B", 
        num_gpus=1 if "72" not in quant else 4)
        for quant in ["0.5", "1.8", "4", "7", "14", "72"]
]

qwen1_5_chat_vllm_models = [
    get_vllm_model(
        abbr=f"qwen1.5-{quant}b-chat", 
        meta_template=qwen_meta_template, 
        num_gpus=1 if "72" not in quant else 4,
        path=f"{ROOT_DIR}models/Qwen/Qwen1.5-{quant}B-Chat")
        for quant in ["0.5", "1.8", "4", "7", "14", "72"]
]