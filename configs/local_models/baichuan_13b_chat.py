from opencompass.models import HuggingFace, HuggingFaceCausalLM

_meta_template = dict(
    round=[
        dict(role='HUMAN', begin='<s> ', end='</s>\n'),
        dict(role='BOT', begin='', end='', generate=True),
    ],
)

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='baichuan-13b-chat',
        path="/mnt/mfs/opsgpt/models/baichuan/Baichuan-13B-Chat",
        tokenizer_path='/mnt/mfs/opsgpt/models/baichuan/Baichuan-13B-Chat',
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              trust_remote_code=True,
                              use_fast=False,),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        meta_template=_meta_template, 
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=2, num_procs=1),
    )
]
