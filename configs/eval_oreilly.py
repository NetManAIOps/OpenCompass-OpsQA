from mmengine.config import read_base
from opencompass.partitioners import SizePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    from .datasets.opsqa.opsqa_demo_summary import opsqa_datasets as summary
    from .datasets.opsqa.opsqa_demo_choice_gen import opsqa_datasets as choice_gen
    from .datasets.opsqa.opsqa_demo_choice_ppl import opsqa_datasets as choice_ppl
    # from .datasets.opsqa.opsqa_demo_compr import opsqa_datasets
    from .datasets.oreilly.oreilly_gen import oreilly_datasets as oreilly
    from .local_models.chatglm_6b import models as chatglm_6b
    from .local_models.chatglm2_6b import models as chatglm2_6b
    from .local_models.baichuan_7b import models as baichuan_7b
    from .models.gpt_3dot5_turbo_peiqi import models as peiqi

datasets = [
    *oreilly[:1] 
    # *summary, 
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

# infer = dict(
#     partitioner=dict(
#         type=SizePartitioner,
#         max_task_size = 100,
#         gen_task_coef = 1,
#     ),
#     runner=dict(
#         type=LocalRunner,
#         max_num_workers=4,
#         task=dict(type=OpenICLInferTask),
#     ),
# )