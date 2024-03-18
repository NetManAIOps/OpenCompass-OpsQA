from opencompass.models import TurboMindModel, LmdeployPytorchModel
from mmengine.config import read_base
with read_base():
    from ...paths import ROOT_DIR
    from ..model_template import get_default_model

qwen1_5_72b_base_lmdeploypytorch = dict(
    type=LmdeployPytorchModel,
    abbr='qwen1.5-72b-base-lmdeploypytorch',
    path=f"{ROOT_DIR}models/Qwen/Qwen1.5-72B-Chat", # 注意路径与huggingface保持一致
    engine_config=dict(session_len=2048,
                        max_batch_size=8,
                        tp=2,
                        thread_safe=True),
    gen_config=dict(top_k=1, top_p=0.8,
                    temperature=0.7,
                    max_new_tokens=400),
    max_out_len=100,
    max_seq_len=2048,
    batch_size=8,
    concurrency=8,
    run_cfg=dict(num_gpus=2, num_procs=1),
)

qwen_72b_chat_turbomind = dict(
    type=TurboMindModel,
    abbr='qwen-72b-chat-turbomind',
    path=f'{ROOT_DIR}models/Qwen/Qwen-72B-Chat',
    engine_config=dict(session_len=2048,
                    max_batch_size=8,
                    tp=2,
                    thread_safe=True),
    gen_config=dict(top_k=1, top_p=0.8,
                max_new_tokens=100),
    max_out_len=400,
    max_seq_len=2048,
    batch_size=8,
    run_cfg=dict(num_gpus=2, num_procs=1),
    meta_template=dict(
        round=[
            dict(role="HUMAN", begin='<|im_start|>user\n', end='<|im_end|>'),
            dict(role="BOT", begin="<|im_start|>assistant\n", end='<|im_end|>', generate=True),
        ],
        )
    )