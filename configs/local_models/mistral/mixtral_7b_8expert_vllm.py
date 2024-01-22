from opencompass.models import VllmAPI

_meta_template = dict(
    round=[
        dict(role="HUMAN", begin='<|im_start|>user\n', end='<|im_end|>'),
        dict(role="BOT", begin="<|im_start|>assistant\n", end='<|im_end|>', generate=True),
    ],
)

models = [
    dict(
        type=VllmAPI,
        abbr='mixtral-7b-8expert',
        path="Mixtral-7B-8Expert",
        url='http://0.0.0.0:8001/generate',
        max_seq_len=2048,
        max_out_len=2048,
        temperature=0,
        batch_size=1,
        meta_template=_meta_template,
    )
]
