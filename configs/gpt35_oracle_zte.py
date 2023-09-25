from mmengine.config import read_base
from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    # Datasets
    from .datasets.oracle.cot import oracle_datasets as oracle_cot
    from .datasets.oracle.naive import oracle_datasets as oracle_naive
    from .datasets.zte.cot import zte_datasets as zte_cot
    from .datasets.zte.naive import zte_datasets as zte_naive
    # Models
    from .models.gpt_4_peiqi import models as gpt_4
    from .models.gpt_3dot5_turbo_peiqi import models as chatgpt

datasets = [
    *oracle_cot, 
    *oracle_naive, 
    *zte_cot, 
    *zte_naive,
]

models = [ 
    # *chatglm2_6b,
    # *qwen_chat_7b,
    # *internlm_chat_7b,
    # *llama2_chat_7b,
    # *baichuan_13b_chat,
    # *llama2_chat_13b,
    # *chinese_llama_2_13b,
    # *chinese_alpaca_2_13b,
    *chatgpt,
    # *gpt_4,
    #*xverse_13b,
]

for model in models:
    # model['path'] = model['path'].replace('/mnt/mfs/opsgpt/','/gpudata/home/cbh/opsgpt/')
    # model['tokenizer_path'] = model['tokenizer_path'].replace('/mnt/mfs/opsgpt/', '/gpudata/home/cbh/opsgpt/')
    # model['run_cfg'] = dict(num_gpus=4, num_procs=1)
    pass

for dataset in datasets:
    # dataset['path'] = dataset['path'].replace('/mnt/mfs/opsgpt/','/gpudata/home/cbh/opsgpt/')
    dataset['sample_setting'] = dict()
    

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
        max_workers_per_gpu=5,
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
