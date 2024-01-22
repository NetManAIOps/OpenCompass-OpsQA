from opencompass.models import CustomModel


models = [
    dict(
        type=CustomModel,
        abbr='AIOPSLLM-WZ',
        path="/mnt/tenant-home_speed/lyh/aiopsllm/models/ops_model.bin",
        # max_out_len=400,
        max_seq_len=2048,
        batch_size=1,
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
    dict(
        type=CustomModel,
        abbr='AIOPSLLM-WZ-QUAN',
        path="/mnt/tenant-home_speed/lyh/aiopsllm/models/ops_model_quant.bin",
        # max_out_len=400,
        max_seq_len=2048,
        batch_size=1,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
