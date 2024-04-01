from mmengine.config import read_base
from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    # Datasets
    from ..datasets.opseval.datasets import all_gen_qa
    # Models
    from ..local_models.baichuan.baichuan import baichuan2_chats_vllm
    from ..local_models.google.gemma import gemma_vllm
    from ..local_models.internlm.internlm import internlm2_chats_vllm
    from ..local_models.lmsys.vicuna import vicuna_bases_vllm
    from ..local_models.mistral.mistral import mistral_7b_vllm
    from ..local_models.qwen.qwen import qwen1_5_chat_vllm_models
    from ..local_models.yi.yi import yi_all_vllm
    from ..models.gpt_3dot5_turbo_peiqi import models as peiqi_models

    from ..paths import ROOT_DIR


datasets = [
    *all_gen_qa
]

# datasets = [datasets[0]]

models = [
    *peiqi_models,
    *baichuan2_chats_vllm,
    *gemma_vllm,
    *internlm2_chats_vllm,
    *vicuna_bases_vllm,
    *yi_all_vllm,
]

for dataset in datasets:
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
