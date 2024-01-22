from opencompass.models import HuggingFace, HuggingFaceCausalLM

qwen_1_8b_base = dict(
        type=HuggingFaceCausalLM,
        abbr='qwen-1.8b-chat',
        path="/mnt/tenant-home_speed/gaozhengwei/models/Qwen/Qwen-1_8B",
        tokenizer_path='/mnt/tenant-home_speed/gaozhengwei/models/Qwen/Qwen-1_8B',
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

