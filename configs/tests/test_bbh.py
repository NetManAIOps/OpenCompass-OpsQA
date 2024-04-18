from mmengine.config import read_base
from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    # Datasets
    from ..datasets.bbh.bbh_gen import bbh_datasets
    from ..datasets.gsm8k.gsm8k_gen import gsm8k_datasets
    from ..datasets.hellaswag.hellaswag_ppl import hellaswag_datasets as hellaswag_ppl_datasets
    from ..datasets.hellaswag.hellaswag_gen import hellaswag_datasets as hellaswag_gen_datasets
    # Models
    from ..local_models.qwen.qwen import qwen1_5_base_models

    from ..paths import ROOT_DIR


datasets = [
    # *bbh_datasets,
    *gsm8k_datasets,
    *hellaswag_gen_datasets,
    *hellaswag_ppl_datasets,
]

# datasets = [datasets[0]]

models = [
    model for model in qwen1_5_base_models if '14' in model['abbr']
]

for dataset in datasets:
    if 'bbh' in dataset['abbr'] or 'gsm8k' in dataset['abbr'] or 'hellaswag' in dataset['abbr']:
        continue
    dataset['sample_setting'] = dict()
    dataset['infer_cfg']['inferencer']['save_every'] = 8
    dataset['infer_cfg']['inferencer']['sc_size'] = 1
    dataset['infer_cfg']['inferencer']['max_token_len'] = 200
    dataset['eval_cfg']['sc_size'] = 1
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