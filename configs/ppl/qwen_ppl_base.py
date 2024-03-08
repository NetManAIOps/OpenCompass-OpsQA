from mmengine.config import read_base
from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    # Datasets
    from ..datasets.ppl_qa.owl_qa import owl_ppl_qa_datasets 
    from ..datasets.ppl_qa.rzy_qa import rzy_ppl_qa_datasets
    from ..datasets.simple_qa.owl_qa import owl_qa_datasets
    from ..datasets.simple_qa.rzy_qa import rzy_qa_datasets
    # Models
    from ..local_models.qwen.qwen import qwen1_5_base_models, qwen1_5_chat_models
datasets = [
    *owl_ppl_qa_datasets,
    *rzy_ppl_qa_datasets,
    *owl_qa_datasets,
    *rzy_qa_datasets
]

models = [
    *qwen1_5_base_models,
]

for model in models:
    pass

for dataset in datasets:
    dataset['sample_setting'] = dict()
    dataset['infer_cfg']['inferencer']['save_every'] = 8
    dataset['infer_cfg']['inferencer']['sc_size'] = 1
    dataset['eval_cfg']['sc_size'] = 1
    if 'network' in dataset['abbr']:
        dataset['sample_setting'] = dict(load_list='/mnt/tenant-home_speed/lyh/evaluation/opseval/network/network_annotated.json')

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
