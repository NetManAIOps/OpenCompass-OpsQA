from opencompass.models import Llama2Chat

ROOT_DIR = '/mnt/mfs/opsgpt/'

_meta_template = dict(
    round=[
        dict(role="HUMAN", api_role="HUMAN"),
        dict(role="BOT", api_role="BOT", generate=True),
    ],
)

models = [
    dict(
        abbr="llama-2-70b-chat",
        type=Llama2Chat,
        path=ROOT_DIR+"models/llama/llama-2-70b-chat/",
        tokenizer_path=ROOT_DIR+"models/llama/tokenizer.model",
        meta_template=_meta_template,
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=8, num_procs=8),
    ),
]