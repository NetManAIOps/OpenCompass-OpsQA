from mmengine.config import read_base
from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    # Datasets
    from ..datasets.opseval.datasets import zte_mc, oracle_mc, owl_mc, owl_qa, network_mc, company_mc, rzy_qa
    # Models
    from ..models.gpt_3dot5_turbo_peiqi import models as chatgpt
    from ..local_models.zhipu.chatglm import chatglm3_6b
    from ..local_models.internlm.internlm import internlm2_chat_20b, internlm2_chat_7b
    from ..local_models.google.gemma import gemma_2b, gemma_7b
    from ..local_models.qwen.qwen import qwen1_5_0_5b_base
    from ..local_models.baichuan.baichuan import baichuan2_turbo, baichuan3
    from ..local_models.zhipu.zhipu import glm_3_turbo, glm_4
    from ..paths import ROOT_DIR

datasets = [
    *zte_mc, *oracle_mc, *owl_mc, *owl_qa, *network_mc, *company_mc, *rzy_qa
]

models = [
    qwen1_5_0_5b_base
]

for model in models:
    model['run_cfg'] = dict(num_gpus=1, num_procs=1)
    pass

for dataset in datasets:
    dataset['sample_setting'] = dict()
    dataset['infer_cfg']['inferencer']['save_every'] = 8
    dataset['infer_cfg']['inferencer']['sc_size'] = 2
    dataset['eval_cfg']['sc_size'] = 2
    dataset['sample_setting'] = dict(sample_size=2)     # !!!WARNING: Use for testing only!!!
    

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
        max_workers_per_gpu=2,
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
