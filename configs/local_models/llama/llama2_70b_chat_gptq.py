from opencompass.models import Llama2Chat, HuggingFaceCausalLM


_meta_template = dict(
    round=[
        dict(role="HUMAN", api_role="HUMAN"),
        dict(role="BOT", api_role="BOT", generate=True),
    ],
)

models = [
    dict(
        abbr="llama-2-70b-chat-gptq",
        type=HuggingFaceCausalLM,
        path="/mnt/tenant-home_speed/liuyuhe/Llama-2-70B-Chat-GPTQ",
        tokenizer_path="/mnt/tenant-home_speed/liuyuhe/Llama-2-70B-Chat-GPTQ",
        meta_template=_meta_template,
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        run_cfg=dict(num_gpus=8, num_procs=8),
    ),
]
