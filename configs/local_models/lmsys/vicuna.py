from mmengine.config import read_base
with read_base():
    from ..model_template import get_default_model
    from ...paths import ROOT_DIR


vicuna_7b_v1_5 = get_default_model(abbr="vicuna-7b-v1.5", path=f"{ROOT_DIR}models/lmsys/vicuna-7b-v1.5")
vicuna_13b_v1_5 = get_default_model(abbr="vicuna-13b-v1.5", path=f"{ROOT_DIR}models/lmsys/vicuna-13b-v1.5")

vicuna_bases = [vicuna_7b_v1_5, vicuna_13b_v1_5]