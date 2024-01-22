from opencompass.models import QwenAPI

ROOT_DIR = './models/'

_meta_template = dict(
    round=[
        dict(role="HUMAN", begin='<|im_start|>user\n', end='<|im_end|>'),
        dict(role="BOT", begin="<|im_start|>assistant\n", end='<|im_end|>', generate=True),
    ],
)

models = [
    dict(
        type=QwenAPI,
        abbr='qwen-72b-chat',
        path="Qwen_Vllm",
        url='http://10.55.33.23:31196/generate',
        max_seq_len=2048,
        max_out_len=2048,
        temperature=0,
        batch_size=1,
        meta_template=_meta_template,
    )
]
