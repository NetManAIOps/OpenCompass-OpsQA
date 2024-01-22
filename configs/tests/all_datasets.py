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
    
    from ..local_models.zhipu.chatglm import chatglm3_6b
    from ..local_models.internlm.internlm import internlm2_chat_20b, internlm2_chat_7b

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
    *company_naive,
]

models = [ 
    *chatglm3_6b,
    *internlm2_chat_20b,
    *internlm2_chat_7b,
]

for model in models:
    # model['path'] = model['path'].replace('/mnt/mfs/opsgpt/','/gpudata/home/cbh/opsgpt/')
    # model['tokenizer_path'] = model['tokenizer_path'].replace('/mnt/mfs/opsgpt/', '/gpudata/home/cbh/opsgpt/')
    # model['run_cfg'] = dict(num_gpus=1, num_procs=1)
    pass

for dataset in datasets:
    dataset['path'] = dataset['path'].replace('/mnt/mfs/opsgpt/evaluation','/mnt/home/opseval/evaluation/')
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
