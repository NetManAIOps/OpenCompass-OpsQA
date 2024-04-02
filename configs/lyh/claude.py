from mmengine.config import read_base
from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    # Datasets
    from ..datasets.opseval.datasets import all_gen_qa, all_gen_mc
    # Models
    from ..local_models.anthropic.claude import claude_3_opus
    # ROOT_DIR
    from ..paths import ROOT_DIR


datasets = [
    *all_gen_mc,
    *all_gen_qa,
]

# Remove COT and 3-shot datasets
datasets = [dataset for dataset in datasets if 'cot' not in dataset['abbr'] and '3-shot' not in dataset['abbr']]

models = [
    claude_3_opus
]

for dataset in datasets:
    dataset['sample_setting'] = dict()
    dataset['infer_cfg']['inferencer']['save_every'] = 8
    dataset['infer_cfg']['inferencer']['sc_size'] = 1
    dataset['infer_cfg']['inferencer']['max_token_len'] = 50
    dataset['eval_cfg']['sc_size'] = 1
    if 'network' in dataset['abbr']:
        dataset['sample_setting'] = dict(load_list=f'{ROOT_DIR}data/opseval/network/network_annotated.json')
    if 'zte' in dataset['abbr']:
        dataset['sample_setting'] = dict(sample_size=500)
    if 'oracle' in dataset['abbr']:
        dataset['sample_setting'] = dict(sample_size=500)
    # dataset['sample_setting'] = dict(sample_size=2)     # !!!WARNING: Use for testing only!!!
    

infer = dict(
    partitioner=dict(
        # type=SizePartitioner,
        # max_task_size=100,
        # gen_task_coef=1,
        type=NaivePartitioner
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=3,
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
