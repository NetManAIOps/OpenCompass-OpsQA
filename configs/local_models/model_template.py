from opencompass.models.huggingface import HuggingFaceCausalLM
from opencompass.models import VLLM


def get_default_model(abbr, path, num_gpus=1, meta_template=None):
    return dict(
        type=HuggingFaceCausalLM,
        abbr=abbr,
        path=path,
        tokenizer_path=path,
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              trust_remote_code=True,
                              use_fast=False,),
        max_out_len=400,
        max_seq_len=2048,
        meta_template=meta_template,
        batch_size=8,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=num_gpus, num_procs=1),
    )

def get_vllm_model(abbr, path, num_gpus=1, meta_template=None, generation_kwargs=dict()):
    return dict(
        type=VLLM,
        abbr=abbr,
        path=path,
        max_seq_len=2048,
        model_kwargs=dict(trust_remote_code=True, max_model_len=2000, tensor_parallel_size=num_gpus, gpu_memory_utilization=0.95),
        generation_kwargs=generation_kwargs,
        meta_template=meta_template,
        max_out_len=400,
        mode='none',
        batch_size=1,
        use_fastchat_template=False,
        run_cfg=dict(num_gpus=num_gpus, num_procs=1),
        end_str=None,
    )