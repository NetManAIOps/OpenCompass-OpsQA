from opencompass.models import HuggingFace, HuggingFaceCausalLM

_meta_template = dict(
    round=[
        dict(role='HUMAN', begin='<|User|>:', end='<eoh>\n'),
        dict(role='BOT', begin='<|Bot|>:', end='', generate=True),
    ],
)

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='internlm_chat_7b',
        path="/mnt/mfs/opsgpt/models/internlm/internlm-chat-7b",
        tokenizer_path='/mnt/mfs/opsgpt/models/internlm/internlm-chat-7b',
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
        run_cfg=dict(num_gpus=2, num_procs=1),
    )
]
