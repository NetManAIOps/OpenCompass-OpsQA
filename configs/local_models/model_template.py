from opencompass.models.huggingface import HuggingFaceCausalLM


def get_default_model(abbr, path):
    return dict(
        type=HuggingFaceCausalLM,
        abbr=abbr,
        path=path,
        tokenizer_path=path,
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