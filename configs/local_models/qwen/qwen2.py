from opencompass.models import HuggingFace, HuggingFaceCausalLM
from mmengine.config import read_base
with read_base():
    from ...paths import ROOT_DIR
    from ..model_template import get_default_model, get_vllm_model
    
qwen_meta_template = dict(
    round=[
        dict(role="HUMAN", begin='<|im_start|>user\n', end='<|im_end|>'),
        dict(role="BOT", begin="<|im_start|>assistant\n", end='<|im_end|>', generate=True),
    ],
)

qwen2_base_models = [
    get_default_model(abbr=f"qwen2-{quant}b-base", 
                      path=f"{ROOT_DIR}models/Qwen/Qwen2-{quant}B", 
                      num_gpus=1 if ("72" not in quant and "57" not in quant) else 4)
        for quant in ["0.5", "1.5", "7", "57B-A14", "72"]
]

qwen2_instruct_models = [
    get_default_model(abbr=f"qwen2-{quant}b-instruct", 
                      meta_template=qwen_meta_template, 
                      num_gpus=1 if ("72" not in quant and "57" not in quant) else 4,
                      path=f"{ROOT_DIR}models/Qwen/Qwen2-{quant}B-Instruct")
        for quant in ["0.5", "1.5", "7", "57B-A14", "72"]
]

qwen2_vllm_models = [
    get_vllm_model(
        abbr=f"qwen2-{quant}b-base", 
        path=f"{ROOT_DIR}models/Qwen/Qwen2-{quant}B", 
        gpu_memory_utilization=0.7 if ("72" not in quant and "57" not in quant) else 0.9,
        num_gpus=1 if ("72" not in quant and "57" not in quant) else 4)
        for quant in ["0.5", "1.5", "7", "57B-A14", "72"]
]

qwen2_instruct_vllm_models = [
    get_vllm_model(
        abbr=f"qwen2-{quant}b-instruct", 
        meta_template=qwen_meta_template, 
        gpu_memory_utilization=0.7 if ("72" not in quant and "57" not in quant) else 0.9,
        num_gpus=1 if ("72" not in quant and "57" not in quant) else 4,
        path=f"{ROOT_DIR}models/Qwen/Qwen2-{quant}B-Instruct")
        for quant in ["0.5", "1.5", "7", "57B-A14", "72"]
]