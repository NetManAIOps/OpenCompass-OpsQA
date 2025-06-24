from mmengine.config import read_base
from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    # Datasets
    from ..datasets.opseval.datasets import company_mc_gen, oracle_mc_gen, network_mc_gen, zte_mc_gen
    # Models
    # from ..local_models.company.chmobile import chmobile
    from ..local_models.nullmodel import models as nullmodel_models
    # ROOT_DIR
    from ..paths import ROOT_DIR


datasets = [
    dataset for dataset in company_mc_gen if 'zjyd' in dataset['abbr'] or 'huaweicloud' in dataset['abbr']
]

datasets += oracle_mc_gen
datasets += network_mc_gen
datasets += zte_mc_gen

datasets = [
    dataset for dataset in datasets if 'multiple' not in dataset['abbr'] and 'cot' not in dataset['abbr']
]

# datasets = [
#     dataset for dataset in datasets if 'zh' in dataset['abbr'].split('-')
# ]

models = [
    *nullmodel_models,
    # *gpt_4_peiqi_models,
    # *gpt_3dot5_turbo_peiqi_models,
    # *qwen2_instruct_vllm_models,
    # *yi_chats_vllm,
    # *baichuan2_chats_vllm,
    # baichuan2_turbo,
    # baichuan3,
    # glm_4,
    # glm_3_turbo,
]


for dataset in datasets:
    dataset['sample_setting'] = dict()
    dataset['infer_cfg']['inferencer']['save_every'] = 8
    dataset['infer_cfg']['inferencer']['sc_size'] = 1
    dataset['infer_cfg']['inferencer']['max_token_len'] = 200
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
        max_num_workers=10,
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
