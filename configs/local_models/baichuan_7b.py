from opencompass.models import HuggingFace, HuggingFaceCausalLM

ROOT_DIR = '/mnt/mfs/opsgpt/'

models = [
    dict(
        type=HuggingFace,
        abbr='baichuan-7b',
        path=ROOT_DIR+"models/baichuan/Baichuan-7B",
        tokenizer_path=ROOT_DIR+'models/baichuan/Baichuan-7B',
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              trust_remote_code=True,
                              use_fast=False,),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=2, num_procs=1),
    )
]
