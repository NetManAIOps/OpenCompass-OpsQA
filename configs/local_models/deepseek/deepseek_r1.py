from opencompass.models import HuggingFace, HuggingFaceCausalLM
from mmengine.config import read_base
with read_base():
    from ...paths import ROOT_DIR
    from ..model_template import get_default_model, get_vllm_model

deepseek_r1_vllm_models = [
    get_vllm_model(
        abbr=f"deepseek-r1-distill-{str.lower(base_quant)}",
        gpu_memory_utilization=0.9,
        num_gpus=1 if '32B' not in base_quant else 2,
        path=f"{ROOT_DIR}models/deepseek-ai/DeepSeek-R1-Distill-{base_quant}")
        for base_quant in ["Llama-8B", "Qwen-1.5B", "Qwen-7B"
                            , "Qwen-14B"
                            , "Qwen-32B"]
]