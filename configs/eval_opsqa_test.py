from mmengine.config import read_base

with read_base():
    from .datasets.opsqa.opsqa_demo_summary import opsqa_datasets as summary
    from .datasets.opsqa.opsqa_demo_choice_gen import opsqa_datasets as choice_gen
    from .datasets.opsqa.opsqa_demo_choice_ppl import opsqa_datasets as choice_ppl
    # from .datasets.opsqa.opsqa_demo_compr import opsqa_datasets
    from .local_models.chatglm_6b import models as chatglm_6b
    from .local_models.baichuan_7b import models as baichuan_7b
    from .models.gpt_3dot5_turbo_peiqi import models as peiqi

datasets = [
    *summary, 
    # *choice_gen, 
    # *choice_ppl
    ]

from opencompass.models import HuggingFaceCausalLM



models = [ 
            *peiqi,
            # *llama2_7b, 
            # *baichuan_7b,
            # *chatglm_6b,
            # *chatglm2_6b
        ]