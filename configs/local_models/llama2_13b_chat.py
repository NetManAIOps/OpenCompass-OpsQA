from opencompass.models import HuggingFaceCausalLM

_meta_template = dict(
    round=[
        dict(role="HUMAN", api_role="HUMAN"),
        dict(role="BOT", api_role="BOT", generate=True),
    ],
)

models = [
    dict(
        abbr="llama-2-13b-chat",
        # type=Llama2Chat,
        type=HuggingFaceCausalLM, 
        path="/mnt/mfs/opsgpt/models/llama-hf/Llama-2-13b-chat-hf/",
        tokenizer_path="/mnt/mfs/opsgpt/models/llama-hf/Llama-2-13b-chat-hf/",
        meta_template=_meta_template,
        max_out_len=20,
        max_seq_len=2048,
        batch_size=16,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=4, num_procs=1),
    ),
]
