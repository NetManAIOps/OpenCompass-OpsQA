from mmengine.config import read_base
from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    # Datasets
    from ..datasets.owl_mc.owl_mc import owl_naive, owl_cot
    # Models
    # from ..local_models.zhipu.zhipu import glm_3_turbo, glm_4
    from ..models.gpt_3dot5_turbo_peiqi import models as chatgpt

datasets = [
    *owl_naive, *owl_cot
]
datasets = [
    dataset for dataset in datasets if 'zh' in dataset['abbr'] and 'en' not in dataset['abbr']
]

models = [
    # glm_3_turbo,
    *chatgpt
]

for model in models:
    pass

for dataset in datasets:
    dataset['sample_setting'] = dict()
    dataset['infer_cfg']['inferencer']['save_every'] = 8
    dataset['infer_cfg']['inferencer']['sc_size'] = 3
    dataset['eval_cfg']['sc_size'] = 3
    if 'network' in dataset['abbr']:
        dataset['sample_setting'] = dict(load_list='/mnt/tenant-home_speed/lyh/evaluation/opseval/network/network_annotated.json')
    dataset['sample_setting']['sample_size'] = 5
    
    

infer = dict(
    partitioner=dict(
        # type=SizePartitioner,
        # max_task_size=100,
        # gen_task_coef=1,
        type=NaivePartitioner
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
