from opencompass.models import HuggingFace

from mmengine.config import read_base
with read_base():
    from ...paths import ROOT_DIR

chatglm3_6b = dict(
        type=HuggingFace,
        abbr='chatglm3-6b',
        path=ROOT_DIR+"models/ZhipuAI/chatglm3-6b",
        tokenizer_path=ROOT_DIR+'models/ZhipuAI/chatglm3-6b',
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              trust_remote_code=True,
                              use_fast=False,),
        max_out_len=400,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )

