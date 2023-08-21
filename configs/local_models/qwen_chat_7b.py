from opencompass.models import QwenLM


models = [
    dict(
        type=QwenLM,
        abbr='qwen_chat_7b',
        path="/mnt/mfs/opsgpt/models/qwen/Qwen-7B-Chat",
        tokenizer_path='/mnt/mfs/opsgpt/models/qwen/Qwen-7B-Chat',
        tokenizer_kwargs=dict(trust_remote_code=True),
        max_out_len=10,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=3, num_procs=1),
    )
]
