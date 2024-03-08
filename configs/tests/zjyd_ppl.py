from mmengine.config import read_base
from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    # Datasets
    from ..datasets.opseval.datasets import zjyd_ppl, zjyd_gen
    # Models
    from ..models.gpt_3dot5_turbo_peiqi import models as chatgpt
    from ..local_models.zhipu.chatglm import chatglm3_6b
    from ..local_models.internlm.internlm import internlm2_chat_20b, internlm2_chat_7b, internlm2_7b
    from ..local_models.google.gemma import gemma_2b, gemma_7b
    from ..local_models.qwen.qwen import qwen1_5_base_models, qwen1_5_chat_models
    from ..local_models.baichuan.baichuan import baichuan2_turbo, baichuan3
    from ..local_models.zhipu.zhipu import glm_3_turbo, glm_4
    from ..paths import ROOT_DIR

datasets = [
    # *zjyd_ppl_datasets
    *zjyd_ppl,
    *zjyd_gen
]

models = [
    *qwen1_5_chat_models,
    *qwen1_5_base_models,
    gemma_2b, gemma_7b,
    internlm2_chat_7b, internlm2_7b,
    chatglm3_6b,
    *chatgpt
]

for model in models:
    model['run_cfg'] = dict(num_gpus=1, num_procs=1)
    pass

for dataset in datasets:
    dataset['sample_setting'] = dict()
    dataset['infer_cfg']['inferencer']['save_every'] = 8
    dataset['infer_cfg']['inferencer']['sc_size'] = 3
    dataset['eval_cfg']['sc_size'] = 3
    # dataset['sample_setting'] = dict(sample_size=2)     # !!!WARNING: Use for testing only!!!
    

infer = dict(
    partitioner=dict(
        # type=SizePartitioner,
        # max_task_size=100,
        # gen_task_coef=1,
        model_first=True,
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
