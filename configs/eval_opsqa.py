from mmengine.config import read_base

with read_base():
    from .datasets.opsqa.opsqa_gen import opsqa_datasets
    from .models.gpt_3dot5_turbo_peiqi import models as peiqi_models
    from .local_models.baichuan_7b import models as baichuan_7b
    from .local_models.chatglm_6b import models as chatglm_6b
    from .local_models.chatglm2_6b import models as chatglm2_6b
    # from .local_models.llama2_7b import models as llama2_7b

datasets = [*opsqa_datasets]

from opencompass.models import HuggingFaceCausalLM



models = [ # *peiqi_models,
            # *llama2_7b, 
            *baichuan_7b,
            *chatglm_6b,
            *chatglm2_6b
        ]