from opencompass.models import HuggingFace, HuggingFaceCausalLM

_meta_template = dict(
    round=[
        dict(role='HUMAN', begin='<|im_start|> ', end='<|im_end|><|endoftext|>\n'),
        dict(role='BOT', begin='<|im_start|> ', end='', generate=True),
    ],
)

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='qwen_chat_7b',
        path="/mnt/mfs/opsgpt/models/qwen/Qwen-7B-Chat",
        tokenizer_path='/mnt/mfs/opsgpt/models/qwen/Qwen-7B-Chat',
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              trust_remote_code=True,
                              use_fast=False,),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        meta_template=_meta_template, 
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        # generate_kwargs=dict(
        #     temperature=0
        # ), 
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
