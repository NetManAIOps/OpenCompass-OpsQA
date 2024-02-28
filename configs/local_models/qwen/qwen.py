from opencompass.models import HuggingFace, HuggingFaceCausalLM

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