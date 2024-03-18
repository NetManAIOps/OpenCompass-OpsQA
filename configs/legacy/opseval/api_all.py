from mmengine.config import read_base
from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    # Datasets
    from ..datasets.company.cot import company_datasets as company_cot
    from ..datasets.company.naive import company_datasets as company_naive
    from ..datasets.network.cot import network_datasets as network_cot
    from ..datasets.network.naive import network_datasets as network_naive
    from ..datasets.oracle.cot import oracle_datasets as oracle_cot
    from ..datasets.oracle.naive import oracle_datasets as oracle_naive
    from ..datasets.owl_mc.cot import owl_datasets as owl_cot
    from ..datasets.owl_mc.naive import owl_datasets as owl_naive
    from ..datasets.zte.cot import zte_datasets as zte_cot
    from ..datasets.zte.naive import zte_datasets as zte_naive
    from ..datasets.simple_qa.tencent import tencent_datasets as tencent_qa
    # Models
    from ..local_models.wenxin import models as wenxin_api

new_datasets = [
    # *company_cot, 
    # *company_naive, 
    # *owl_cot,
    # *owl_naive,
    *network_cot,
    *network_naive,
    *oracle_cot,
    *oracle_naive,
    *zte_cot,
    *zte_naive,
    # *tencent_qa,
]

models = [ 
    *wenxin_api, 
]

for model in models:
    # model['path'] = model['path'].replace('/mnt/mfs/opsgpt/','/gpudata/home/cbh/opsgpt/')
    # model['tokenizer_path'] = model['tokenizer_path'].replace('/mnt/mfs/opsgpt/', '/gpudata/home/cbh/opsgpt/')
    # model['run_cfg'] = dict(num_gpus=4, num_procs=1)
    pass

for dataset in new_datasets:
    # dataset['path'] = dataset['path'].replace('/mnt/mfs/opsgpt/','/gpudata/home/cbh/opsgpt/')
    dataset['infer_cfg']['inferencer']['sc_size'] = 1
    dataset['infer_cfg']['inferencer']['save_every'] = 1
    dataset['eval_cfg']['sc_size'] = 1
    dataset['sample_setting'] = dict(sample_size=50)
    dataset['infer_cfg']['inferencer']['fix_id_list'] = [0]

datasets = []
for dataset in new_datasets:
    if 'multiple' not in dataset['abbr'] and 'ero' not in dataset['abbr']:
        datasets.append(dataset)
    

infer = dict(
    partitioner=dict(
        # type=SizePartitioner,
        # max_task_size=100,
        # gen_task_coef=1,
        type=NaivePartitioner
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=2,
        max_workers_per_gpu=0,
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
