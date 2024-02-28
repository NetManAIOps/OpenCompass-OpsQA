from opencompass.models import HuggingFace, HuggingFaceCausalLM
from mmengine.config import read_base



nm_qwen_14b_bookall_lora= dict(
        type=HuggingFaceCausalLM,
        abbr='nm_qwen_14b_bookall_lora',
        # /mnt/tenant-home_speed/lyh/OpenCompass-OpsQA-lyh/models/xll_models/Qwen-1_8B/lora/Qwen-14B-bookall-lora
        path=f"/mnt/tenant-home_speed/lyh/OpenCompass-OpsQA-lyh/models/xll_models/Qwen-1_8B/books_all-lora",
        tokenizer_path=f"/mnt/tenant-home_speed/lyh/OpenCompass-OpsQA-lyh/models/xll_models/Qwen-1_8B/books_all-lora",
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              trust_remote_code=True,
                              use_fast=False,),
        max_out_len=400,
        max_seq_len=2048,
        batch_size=1,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )