from mmengine.config import read_base
from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    # Datasets
    from .datasets.oreilly.fewshot_sc_cot import oreilly_datasets as fewshot_sc_cot
    from .datasets.oreilly.zeroshot_sc_cot import oreilly_datasets as zeroshot_sc_cot
    from .datasets.network_zh.fewshot_sc_cot import oreilly_datasets as zh_fewshot_sc_cot
    from .datasets.network_zh.zeroshot_sc_cot import oreilly_datasets as zh_zeroshot_sc_cot
    # Models
    from .models.gpt_4_peiqi import models as gpt_4

datasets = [
    # *fewshot_naiive, # ok
    *fewshot_sc_cot,
    # *zeroshot_sc_cot,
    *zh_fewshot_sc_cot,
    # *zh_zeroshot_sc_cot,
]

models = [ 
    *gpt_4,
]

for model in models:
    # model['path'] = model['path'].replace('/mnt/mfs/opsgpt/','/gpudata/home/cbh/opsgpt/')
    # model['tokenizer_path'] = model['tokenizer_path'].replace('/mnt/mfs/opsgpt/', '/gpudata/home/cbh/opsgpt/')
    # if '13' in model['path']:
    #     model['run_cfg'] = dict(num_gpus=4, num_procs=1)
    # else:
    #     model['run_cfg'] = dict(num_gpus=4, num_procs=8)
    # pass
    pass

for dataset in datasets:
    # dataset['path'] = dataset['path'].replace('/mnt/mfs/opsgpt/','/gpudata/home/cbh/opsgpt/')
    #dataset['sample_setting'] = dict(sample_size=1)
    dataset['infer_cfg']['inferencer']['sc_size'] = 1
    dataset['infer_cfg']['inferencer']['save_every'] = 1
    dataset['eval_cfg']['sc_size'] = 1
    if 'zh' in dataset['abbr']:
        dataset['sample_setting'] = dict(load_list='/mnt/mfs/opsgpt/opencompass/experiments/gpt4/chatgpt_fail_zh.json')
    else:
        dataset['sample_setting'] = dict(load_list='/mnt/mfs/opsgpt/opencompass/experiments/gpt4/chatgpt_fail_en.json')

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
