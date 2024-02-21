from mmengine.config import read_base
from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    # Commons
    from ..paths import ROOT_DIR
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
    from ..local_models.zhipu.zhipu import glm_3_turbo, glm_4
    from ..local_models.baichuan.baichuan import baichuan2_turbo, baichuan3

datasets = [
    *company_cot, *company_naive, 
    *network_cot, *network_naive, 
    *zte_cot, *zte_naive, 
    *oracle_cot, *oracle_naive, 
    *owl_cot, *owl_naive, 
]
# datasets = [
#     dataset for dataset in datasets if 'Zero-shot' in dataset['abbr'] and 'cot' not in dataset['abbr'] and 'zh' in dataset['abbr'] and 'en' not in dataset['abbr']
# ]

models = [ 
    # internlm2_chat_7b,
    # *chatgpt,
    # glm_3_turbo,
    # glm_4,
    baichuan2_turbo,
    # baichuan3,
]

for model in models:
    # model['path'] = model['path'].replace('/mnt/mfs/opsgpt/','/gpudata/home/cbh/opsgpt/')
    # model['tokenizer_path'] = model['tokenizer_path'].replace('/mnt/mfs/opsgpt/', '/gpudata/home/cbh/opsgpt/')
    # model['run_cfg'] = dict(num_gpus=1, num_procs=1)
    pass

for dataset in datasets:
    # dataset['path'] = dataset['path'].replace('/mnt/mfs/opsgpt/evaluation','/mnt/home/opseval/evaluation/')
    dataset['sample_setting'] = dict()
    # dataset['infer_cfg']['inferencer']['save_every'] = 8
    dataset['infer_cfg']['inferencer']['sc_size'] = 1
    dataset['eval_cfg']['sc_size'] = 1
    if 'network' in dataset['abbr']:
        dataset['sample_setting'] = dict(load_list=f'{ROOT_DIR}data/opseval/network/network_annotated.json')
    dataset['sample_setting']['sample_size'] = 200
    
    

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
