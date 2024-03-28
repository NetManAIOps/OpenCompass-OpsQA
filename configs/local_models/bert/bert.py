from opencompass.models import Bert

from mmengine.config import read_base
with read_base():
    from ...paths import ROOT_DIR

bert_large_cased = dict(
        type=Bert,
        abbr='bert_large_cased',
        path=ROOT_DIR+"models/google-bert/bert-large-cased",
        tokenizer_path=ROOT_DIR+"models/google-bert/bert-large-cased",
        tokenizer_kwargs=dict(trust_remote_code=True),
        max_out_len=400,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(trust_remote_code=True),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )