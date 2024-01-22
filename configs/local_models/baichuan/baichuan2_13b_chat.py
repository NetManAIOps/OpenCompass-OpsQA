from opencompass.models import HuggingFaceCausalLM

ROOT_DIR = '/mnt/tenant-home_speed/gaozhengwei/'

_meta_template = dict(
    round=[
        dict(role="HUMAN", api_role="HUMAN"),
        dict(role="BOT", api_role="BOT", generate=True),
    ],
)

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='baichuan2-13b-chat',
        path=ROOT_DIR+"models/baichuan-inc/Baichuan2-13B-Chat",
        tokenizer_path=ROOT_DIR+'models/baichuan-inc/Baichuan2-13B-Chat',
        tokenizer_kwargs=dict(trust_remote_code=True,
                              use_fast=False,),
        meta_template=_meta_template,
        max_out_len=1024,
        max_seq_len=2048,
        batch_size=16, 
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=4, num_procs=1),
    )
]
