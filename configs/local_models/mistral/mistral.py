from mmengine.config import read_base
with read_base():
    from ..model_template import get_default_model, get_vllm_model
    from ...paths import ROOT_DIR

mistral_7b = get_default_model(abbr="mistral-7b", path=f"{ROOT_DIR}models/mistralai/Mistral-7B-v0.1")

mistral_7b_vllm = get_vllm_model(abbr="mistral-7b", path=f"{ROOT_DIR}models/mistralai/Mistral-7B-v0.1")