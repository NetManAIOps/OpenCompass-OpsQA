from mmengine.config import read_base
from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    from .datasets.opsqa.opsqa import opsqa_datasets as opsqa
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
    *opsqa,
]
models = [
    # *llama2_13b,
    # *baichuan_13b,
    # *xverse_13b,
    # *chinese_llama_2_13b,
    # *chinese_alpaca_2_13b,
    # *chatglm2_6b,
    # *internlm_7b,
    *qwen_chat_7b,
    # *chatgpt
]

for model in models:
    model['max_out_len'] = 2048
    if 'GPT' not in model['abbr']:
        model['run_cfg'] = dict(num_gpus=2, num_procs=1) if '7b' in model['abbr'] or '6b' in model['abbr'] else dict(num_gpus=4, num_procs=1)
        model['batch_size'] = 16

infer = dict(
    partitioner=dict(
        type=SizePartitioner,
        max_task_size=16,
        gen_task_coef=1,
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=16,
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