from mmengine.config import read_base
from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    # Datasets
    from ..datasets.opseval.datasets import zte_mc, oracle_mc, owl_mc, owl_qa, network_mc, company_mc, rzy_qa
    # Models
    from ..local_models.qwen.turbomind import qwen1_5_72b_base_lmdeploypytorch, qwen_72b_chat_turbomind
    from ..paths import ROOT_DIR

datasets = [
    *owl_mc
]

models = [
    qwen1_5_72b_base_lmdeploypytorch,
    qwen_72b_chat_turbomind,
]

for dataset in datasets:
    dataset['sample_setting'] = dict()
    dataset['infer_cfg']['inferencer']['save_every'] = 8
    dataset['infer_cfg']['inferencer']['sc_size'] = 2
    dataset['eval_cfg']['sc_size'] = 2
    dataset['sample_setting'] = dict(sample_size=100)     # !!!WARNING: Use for testing only!!!
    

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
