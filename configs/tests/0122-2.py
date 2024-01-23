from mmengine.config import read_base
from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    # Datasets
    from ..datasets.company.company import company_cot, company_naive
    from ..datasets.network.network import network_cot, network_naive
    from ..datasets.zte.zte import zte_cot, zte_naive
    from ..datasets.oracle.oracle import oracle_cot, oracle_naive
    from ..datasets.owl_mc.owl_mc import owl_cot, owl_naive
    # Models
    from ..models.gpt_3dot5_turbo_peiqi import models as chatgpt
    from ..local_models.zhipu.chatglm import chatglm3_6b
    from ..local_models.internlm.internlm import internlm2_chat_20b, internlm2_chat_7b

datasets = [
    *company_cot, *company_naive, 
    *network_cot, *network_naive, 
    *zte_cot, *zte_naive, 
    *oracle_cot, *oracle_naive, 
    *owl_cot, *owl_naive, 
]

models = [ 
    internlm2_chat_20b,
    # *chatgpt
]

for model in models:
    # model['path'] = model['path'].replace('/mnt/mfs/opsgpt/','/gpudata/home/cbh/opsgpt/')
    # model['tokenizer_path'] = model['tokenizer_path'].replace('/mnt/mfs/opsgpt/', '/gpudata/home/cbh/opsgpt/')
    model['run_cfg'] = dict(num_gpus=2, num_procs=1)
    pass

for dataset in datasets:
    # dataset['path'] = dataset['path'].replace('/mnt/mfs/opsgpt/evaluation','/mnt/home/opseval/evaluation/')
    # dataset['sample_setting'] = dict(sample_size=1)
    dataset['sample_setting'] = dict()
    dataset['infer_cfg']['inferencer']['save_every'] = 8
    dataset['infer_cfg']['inferencer']['sc_size'] = 3
    dataset['eval_cfg']['sc_size'] = 3
    if 'network' in dataset['abbr']:
        dataset['sample_setting'] = dict(load_list=f'{ROOT_DIR}data/opseval/network/network_annotated.json')
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
