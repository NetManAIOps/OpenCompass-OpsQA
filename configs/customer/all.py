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
    from ..local_models.aiopsllm import models as aiopsllm

datasets = [
    *network_cot, 
    *network_naive, 
    *zte_cot,
    *zte_naive,
    *oracle_cot,
    *oracle_naive,
    *owl_cot,
    *owl_naive,
    *company_cot,
    *company_naive
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
    # *chatgpt,
    # *gpt_4,
    #*xverse_13b,
    # *devops_model_14b_base,
    *aiopsllm,
]

for model in models:
    # model['path'] = model['path'].replace('/mnt/mfs/opsgpt/','/gpudata/home/cbh/opsgpt/')
    # model['tokenizer_path'] = model['tokenizer_path'].replace('/mnt/mfs/opsgpt/', '/gpudata/home/cbh/opsgpt/')
    model['run_cfg'] = dict(num_gpus=1, num_procs=1)
    pass

for dataset in datasets:
    dataset['path'] = dataset['path'].replace('/mnt/mfs/opsgpt/evaluation','/mnt/home/opseval/evaluation/')
    dataset['sample_setting'] = dict()
    dataset['infer_cfg']['inferencer']['sc_size'] = 3
    dataset['infer_cfg']['inferencer']['save_every'] = 3
    if 'zte' in dataset['abbr']:
        dataset['sample_setting'] = dict(sample_size=500)
    if 'network' in dataset['abbr']:
        dataset['sample_setting'] = dict(load_list='/mnt/home/opseval/evaluation/opseval/network/network_annotated.json', sample_size=500)

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
        max_workers_per_gpu=3,
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
