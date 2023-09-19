from opencompass.models import Llama2

models = [
    dict(
        abbr="llama-2-70b",
        type=Llama2Chat,
        path="/gpudata/home/cbh/opsgpt/models/llama/llama-2-70b/",
        tokenizer_path="/gpudata/home/cbh/opsgpt/models/llama/tokenizer.model",
        meta_template=_meta_template,
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=8, num_procs=8),
    ),
]