from mmengine.config import read_base
from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    # Datasets
    from ..datasets.network.cot import network_datasets as network_cot
    from ..datasets.network.naive import network_datasets as network_naive
    from ..datasets.zte.cot import zte_datasets as zte_cot
    from ..datasets.zte.naive import zte_datasets as zte_naive
    from ..datasets.oracle.cot import oracle_datasets as oracle_cot
    from ..datasets.oracle.naive import oracle_datasets as oracle_naive
    from ..datasets.owl_mc.cot import owl_datasets as owl_cot
    from ..datasets.owl_mc.naive import owl_datasets as owl_naive
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
    
    from ..local_models.devops_model_14b_base import models as devops_model_14b_base
    from ..local_models.qwen_14b_chat import models as qwen_14b_chat
    from ..local_models.qwen_72b_chat import models as qwen_72b_chat
    
    from ..local_models.aquilachat2_34b import models as aquilachat2_34b
    #from ..local_models.yi_34b import models as yi_34b
    from ..local_models.chatglm3_6b import models as chatglm3_6b
    from ..local_models.baichuan2_13b_chat import models as baichuan2_13b_chat
    from ..local_models.mistral_7b import models as mistral_7b

datasets = [
    *network_cot, 
    *network_naive, 
    #*zte_cot,
    #*zte_naive,
    #*oracle_cot,
    #*oracle_naive,
    #*owl_cot,
    #*owl_naive,
    *company_cot,
    *company_naive,
]

models = [ 
    # *chatglm2_6b,
    # *qwen_chat_7b,
    # *internlm_chat_7b,
    # *baichuan_13b_chat,
    # *chinese_llama_2_13b,
    # *chinese_alpaca_2_13b,
    # *chatgpt,
    # *gpt_4,
    #*xverse_13b,
    *qwen_72b_chat,
    # *llama2_chat_7b,
    # *baichuan2_13b_chat,
    # *chatglm3_6b,
    # *devops_model_14b_base,
    # *llama2_chat_13b,    # not ready, we don't have the model
    # *mistral_7b
    #*yi_34b,
    #*aquilachat2_34b,
]

for model in models:
    # model['path'] = model['path'].replace('/mnt/mfs/opsgpt/','/gpudata/home/cbh/opsgpt/')
    # model['tokenizer_path'] = model['tokenizer_path'].replace('/mnt/mfs/opsgpt/', '/gpudata/home/cbh/opsgpt/')
    model['run_cfg'] = dict(num_gpus=2, num_procs=1)
    pass

for dataset in datasets:
    dataset['path'] = dataset['path'].replace('/mnt/mfs/opsgpt/evaluation','/mnt/home/opseval/evaluation/')
    dataset['sample_setting'] = dict()
    if 'zte' in dataset['abbr']:
        dataset['sample_setting'] = dict(sample_size=500)
    

infer = dict(
    partitioner=dict(
        # type=SizePartitioner,
        # max_task_size=100,
        # gen_task_coef=1,
        type=NaivePartitioner
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=32,
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
