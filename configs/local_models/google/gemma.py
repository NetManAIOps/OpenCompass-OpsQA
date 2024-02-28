from opencompass.models import HuggingFaceCausalLM

from mmengine.config import read_base
with read_base():
    from ...paths import ROOT_DIR

gemma_2b = dict(
        type=HuggingFaceCausalLM,
        abbr='gemma_2b',
        path=ROOT_DIR+"models/google/gemma-2b",
        tokenizer_path=ROOT_DIR+"models/google/gemma-2b",
        tokenizer_kwargs=dict(trust_remote_code=True),
        max_out_len=400,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )

gemma_7b = dict(
        type=HuggingFaceCausalLM,
        abbr='gemma_7b',
        path=ROOT_DIR+"models/google/gemma-7b",
        tokenizer_path=ROOT_DIR+"models/google/gemma-7b",
        tokenizer_kwargs=dict(trust_remote_code=True),
        max_out_len=400,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )