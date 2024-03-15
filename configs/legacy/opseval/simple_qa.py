from mmengine.config import read_base
from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    # Datasets
    from ..datasets.simple_qa.tencent import tencent_datasets as tencent_qa 
    from ..datasets.simple_qa.zte import zte_datasets as zte_qa
    from ..datasets.simple_qa.rzy import rzy_datasets as rzy_qa 
    # Models
    from ..local_models.chatglm3_6b import models as chatglm3_6b
    from ..local_models.qwen_chat_7b import models as qwen_chat_7b
    from ..local_models.qwen_14b_chat import models as qwen_14b_chat
    from ..local_models.baichuan2_13b_chat import models as baichuan2_13b_chat
    from ..local_models.llama2_7b_chat import models as llama2_chat_7b
    from ..local_models.llama2_13b_chat import models as llama2_chat_13b
    from ..local_models.devops_model_14b_chat import models as devops_model_14b_chat
    from ..local_models.yi_34b_chat import models as yi_34b_chat
    from ..local_models.mistral_7b import models as mistral_7b

datasets = [
    *tencent_qa,
    *rzy_qa,
    *zte_qa,
]

models = [ 
    *chatglm3_6b,
    *qwen_14b_chat,
    # *internlm_chat_7b,
    *llama2_chat_7b,
    # *baichuan2_13b_chat,
    # *llama2_chat_13b,
    #*qwen_chat_7b,
    # *devops_model_14b_chat,
    # *yi_34b_chat,
    # *mistral_7b,
    # *chinese_llama_2_13b,
    # *chinese_alpaca_2_13b,
    # *chatgpt,
    # *gpt_4,
    #*xverse_13b,
]

for model in models:
    # model['path'] = model['path'].replace('/mnt/mfs/opsgpt/','/gpudata/home/cbh/opsgpt/')
    # model['tokenizer_path'] = model['tokenizer_path'].replace('/mnt/mfs/opsgpt/', '/gpudata/home/cbh/opsgpt/')
    model['run_cfg'] = dict(num_gpus=1, num_procs=1)
    model['max_out_len']=1024
    pass

for dataset in datasets:
    dataset['path'] = dataset['path'].replace('/mnt/mfs/opsgpt/','/mnt/home/opseval/')
    dataset['infer_cfg']['inferencer']['save_every'] = 1
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
