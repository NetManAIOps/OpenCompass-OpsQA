from mmengine.config import read_base
from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    # Datasets
    from ..datasets.opseval.datasets import all_ppl_mc, all_ppl_qa
    # Models
    from ..local_models.qwen.qwen import qwen1_5_base_models
    from ..local_models.lmsys.vicuna import vicuna_bases
    from ..local_models.yi.yi import yi_bases
    from ..local_models.mistral.mistral import mistral_7b
    from ..local_models.internlm.internlm import internlm2_bases
    from ..local_models.google.t5 import t5_base
    # Paths
    from ..paths import ROOT_DIR

datasets = [
    *all_ppl_mc,
    *all_ppl_qa,
]

models = [
    t5_base, 
    *qwen1_5_base_models,
    *vicuna_bases,
    *yi_bases,
    mistral_7b,
    *internlm2_bases,
]

models = [
    model for model in models if '34b' not in model['abbr']
]

for model in models:
    pass

for dataset in datasets:
    dataset['sample_setting'] = dict()
    dataset['infer_cfg']['inferencer']['save_every'] = 8
    dataset['infer_cfg']['inferencer']['sc_size'] = 1
    dataset['eval_cfg']['sc_size'] = 1
    if 'network' in dataset['abbr']:
        dataset['sample_setting'] = dict(load_list=f'{ROOT_DIR}data/opseval/network/network_annotated.json')

infer = dict(
    partitioner=dict(
        # type=SizePartitioner,
        # max_task_size=100,
        # gen_task_coef=1,
        type=NaivePartitioner,
        model_first=True,
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
