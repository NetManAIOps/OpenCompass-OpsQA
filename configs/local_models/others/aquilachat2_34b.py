from opencompass.models import HuggingFace, HuggingFaceCausalLM


models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='aquilachat2_34b',
        path="/mnt/tenant-home_speed/gaozhengwei/models/BAAI/AquilaChat2-34B",
        tokenizer_path="/mnt/tenant-home_speed/gaozhengwei/models/BAAI/AquilaChat2-34B",
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              trust_remote_code=True,
                              use_fast=False,),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=2, num_procs=1),
    )
]