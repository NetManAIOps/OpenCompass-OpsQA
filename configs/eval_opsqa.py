from mmengine.config import read_base

with read_base():
    from .datasets.opsqa.opsqa_gen import opsqa_datasets
    from .models.gpt_3dot5_turbo_peiqi import models as peiqi_models
    from .models.hf_baichuan_7b import models as baichuan_7b
    from .models.hf_chatglm_6b import models as chatglm_6b
    from .models.hf_falcon_7b import models as falcon_7b
    from .models.hf_internlm_7b import models as internlm_7b
    from .models.hf_llama_7b import models as llama_7b
    from .models.hf_llama2_7b import models as llama2_7b
    from .models.hf_vicuna_7b import models as vicuna_7b

datasets = [*opsqa_datasets]

from opencompass.models import HuggingFaceCausalLM



models = [ # *peiqi_models, 
            *baichuan_7b,
            *chatglm_6b,
            *falcon_7b,
            *internlm_7b,
            *llama_7b, 
            *llama2_7b,
            *vicuna_7b]