from opencompass.models import HuggingFaceCausalLM

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='internlm_chat_7b',
        path="/gpudata/home/cbh/opsgpt/models/internlm/internlm-chat-7b",
        tokenizer_path='/gpudata/home/cbh/opsgpt/models/internlm/internlm-chat-7b',
        tokenizer_kwargs=dict(trust_remote_code=True),
        max_out_len=1024,
        max_seq_len=2048,
        batch_size=16,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=4, num_procs=1),
    )
]
