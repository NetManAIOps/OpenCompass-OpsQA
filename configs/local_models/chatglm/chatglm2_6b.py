from opencompass.models import HuggingFace

ROOT_DIR = '/mnt/mfs/opsgpt/'

models = [
    dict(
        type=HuggingFace,
        abbr='chatglm2-6b',
        path=ROOT_DIR+"models/chatglm/chatglm2-6b",
        tokenizer_path=ROOT_DIR+'models/chatglm/chatglm2-6b',
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              trust_remote_code=True,
                              use_fast=False,),
        max_out_len=1024,
        max_seq_len=2048,
        batch_size=16,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        # generate_kwargs=dict(
        #     temperature=0
        # ), 
        run_cfg=dict(num_gpus=4, num_procs=1),
    )
]
