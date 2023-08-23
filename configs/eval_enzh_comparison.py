from mmengine.config import read_base
from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    # Datasets
    from .datasets.oreilly.zh_en_comparison import oreilly_datasets as comparison
    # Models
    from .local_models.chatglm_6b import models as chatglm_6b
    from .local_models.chatglm2_6b import models as chatglm2_6b
    from .local_models.qwen_chat_7b import models as qwen_chat_7b
    from .local_models.baichuan_7b import models as baichuan_7b
    from .local_models.internlm_chat_7b import models as internlm_chat_7b
    from .local_models.llama2_7b_chat import models as llama2_chat_7b
    from .local_models.chinese_llama_2_13b import models as chinese_llama_2_13b
    from .local_models.chinese_alpaca_2_13b import models as chinese_alpaca_2_13b
    from .models.gpt_4_peiqi import models as gpt_4
    from .models.gpt_3dot5_turbo_peiqi import models as chatgpt

datasets = [
    *comparison
]

models = [ 
    # *chatgpt,
    *chinese_llama_2_13b,
    *chinese_alpaca_2_13b,
]

for ds in datasets:
    pass
    # ds['sample_setting'] = dict(sample_size=1)

for model in models:
    model['max_out_len'] = 100
    model['run_cfg'] = dict(num_gpus=3, num_procs=1)

infer = dict(
    partitioner=dict(
        # type=SizePartitioner,
        # max_task_size=100,
        # gen_task_coef=1,
        type=NaivePartitioner
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=32,
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