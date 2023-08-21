from opencompass.models import HuggingFaceCausalLM

_meta_template = dict(
    round=[
        dict(role="HUMAN", api_role="HUMAN"),
        dict(role="BOT", api_role="BOT", generate=True),
    ],
)

models = [
    dict(
        abbr="chinese-alpaca-2-13b",
        type=HuggingFaceCausalLM, 
        path="/mnt/mfs/opsgpt/models/chinese-llama-2/chinese-alpaca-2-13b/",
        tokenizer_path="/mnt/mfs/opsgpt/models/chinese-llama-2/chinese-alpaca-2-13b/",
        max_out_len=20,
        max_seq_len=2048,
        batch_size=16,
        meta_template=_meta_template,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        batch_padding=False,
        run_cfg=dict(num_gpus=4, num_procs=1),
    ),
]
