from mmengine.config import read_base
from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    from .datasets.oreilly.oreilly_few_shot import oreilly_datasets as oreilly
    from .local_models.chatglm_6b import models as chatglm_6b
    from .local_models.chatglm2_6b import models as chatglm2_6b
    from .local_models.qwen_chat_7b import models as qwen_chat_7b
    from .local_models.baichuan_7b import models as baichuan_7b
    from .local_models.internlm_chat_7b import models as internlm_7b
    from .local_models.llama2_13b_chat import models as llama2_13b
    from .local_models.chinese_llama_2_13b import models as chinese_llama_2_13b
    from .local_models.chinese_alpaca_2_13b import models as chinese_alpaca_2_13b
    from .local_models.baichuan_13b_chat import models as baichuan_13b
    from .local_models.xverse_13b import models as xverse_13b
    from .models.gpt_4_peiqi import models as gpt_4
    from .models.gpt_3dot5_turbo_peiqi import models as chatgpt

datasets = [
    *oreilly,
]
models = [
    # *llama2_13b
    *baichuan_13b,
    # *xverse_13b,
    # *internlm_7b,
    # *qwen_chat_7b,
    # *chinese_llama_2_13b,
    # *chinese_alpaca_2_13b,
    # *llama2_7b, 
    # *baichuan_7b,
    # *chatglm_6b,
    # *chatglm2_6b,
    # *chatgpt
]

infer = dict(
    partitioner=dict(
        type=SizePartitioner,
        max_task_size=1500,
        gen_task_coef=1,
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=1,
        max_workers_per_gpu=1,
        task=dict(type=OpenICLInferTask),
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=32,
        task=dict(type=OpenICLEvalTask)),
)