from mmengine.config import read_base
from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    # Datasets
    from ..datasets.opseval.datasets import owl_mc, owl_qa
    # Models
    from ..local_models.qwen.qwen import qwen1_5_4b_base
    from ..paths import ROOT_DIR

datasets = [
    *owl_mc, *owl_qa,
]


models = [
    qwen1_5_4b_base
]

for dataset in datasets:
    dataset['sample_setting'] = dict()
    dataset['infer_cfg']['inferencer']['save_every'] = 8
    dataset['infer_cfg']['inferencer']['sc_size'] = 3
    dataset['infer_cfg']['inferencer']['max_token_len'] = 20
    dataset['eval_cfg']['sc_size'] = 3
    dataset['sample_setting'] = dict(sample_size=10)     # !!!WARNING: Use for testing only!!!
    

infer = dict(
    partitioner=dict(
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
