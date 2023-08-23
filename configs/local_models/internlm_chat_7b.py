from opencompass.models import HuggingFaceCausalLM

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='internlm_chat_7b',
        path="/mnt/mfs/opsgpt/models/internlm/internlm-chat-7b",
        tokenizer_path='/mnt/mfs/opsgpt/models/internlm/internlm-chat-7b',
        tokenizer_kwargs=dict(trust_remote_code=True),
        max_out_len=20,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=2, num_procs=1),
    )
]
