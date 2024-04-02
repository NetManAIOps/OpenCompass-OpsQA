from mmengine.config import read_base
from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    # Datasets
    from ..datasets.opseval.datasets import owl_mc, owl_qa
    # Models
    from ..local_models.google.t5 import t5_base
    from ..local_models.bert.bert import bert_large_cased
    from ..local_models.qwen.qwen_vllm import qwen_1_5b_14b_chat_vllm

    from ..paths import ROOT_DIR


datasets = [
    *owl_mc, *owl_qa,
]

datasets = [
    dataset for dataset in datasets if 'Zero-shot' in dataset['abbr'] and 'zh' in dataset['abbr']
]

models = [
    t5_base,
    # bert_large_cased,
    qwen_1_5b_14b_chat_vllm,
    # *vicuna_bases,
    # *internlm2_bases,
    # *yi_bases, 
    # mistral_7b
]

for model in models:
    model['run_cfg'] = dict(num_gpus=1, num_procs=1)
    pass

for dataset in datasets:
    dataset['sample_setting'] = dict()
    dataset['infer_cfg']['inferencer']['save_every'] = 8
    dataset['infer_cfg']['inferencer']['sc_size'] = 2
    dataset['infer_cfg']['inferencer']['max_token_len'] = 20
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
