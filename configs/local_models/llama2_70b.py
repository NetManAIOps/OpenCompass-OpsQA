from opencompass.models import Llama2

ROOT_DIR = '/mnt/mfs/opsgpt/'

models = [
    dict(
        abbr="llama-2-70b",
        type=Llama2,
        path=ROOT_DIR+"models/llama/llama-2-70b/",
        tokenizer_path=ROOT_DIR+"models/llama/tokenizer.model",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=8, num_procs=8),
    ),
]