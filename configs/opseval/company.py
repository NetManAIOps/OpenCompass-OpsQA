from mmengine.config import read_base
from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    # Datasets
    from ..datasets.company.cot import company_datasets as company_cot
    from ..datasets.company.naive import company_datasets as company_naive
    # Models
    from ..local_models.chatglm2_6b import models as chatglm2_6b
    from ..local_models.qwen_chat_7b import models as qwen_chat_7b
    from ..local_models.baichuan_13b_chat import models as baichuan_13b_chat
    from ..local_models.internlm_chat_7b import models as internlm_chat_7b
    from ..local_models.llama2_7b_chat import models as llama2_chat_7b
    from ..local_models.llama2_13b_chat import models as llama2_chat_13b
    from ..local_models.chinese_llama_2_13b import models as chinese_llama_2_13b
    from ..local_models.chinese_alpaca_2_13b import models as chinese_alpaca_2_13b
    from ..local_models.xverse_13b import models as xverse_13b
    from ..models.gpt_4_peiqi import models as gpt_4
    from ..models.gpt_3dot5_turbo_peiqi import models as chatgpt

datasets = [
    *company_cot, 
    *company_naive, 
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
