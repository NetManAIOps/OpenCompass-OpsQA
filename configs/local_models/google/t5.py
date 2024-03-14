from opencompass.models import T5

from mmengine.config import read_base
with read_base():
    from ...paths import ROOT_DIR

t5_base = dict(
        type=T5,
        abbr='t5_base',
        path=ROOT_DIR+"models/google-t5/t5-base",
        tokenizer_path=ROOT_DIR+"models/google-t5/t5-base",
        tokenizer_kwargs=dict(trust_remote_code=True),
        max_out_len=400,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )