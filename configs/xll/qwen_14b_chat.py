from mmengine.config import read_base
from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    # Datasets
    from ..datasets.company.company import company_cot, company_naive
    from ..datasets.network.network import network_cot, network_naive
    from ..datasets.zte.zte_ppl import zte_naive as zte_naive_ppl
    from ..datasets.zte.zte import zte_naive, zte_cot
    from ..datasets.oracle.oracle import oracle_cot, oracle_naive
    from ..datasets.owl_mc.owl_mc import owl_cot, owl_naive
    
    #/mnt/tenant-home_speed/lyh/OpenCompass-OpsQA-lyh/models/xll_models/Qwen-1_8B/mixture/mix5_1-sft_owl-model
    # Models
    # from ..netman_models.qwen import nm_qwen_14b_bookall_lora as zzr1
    # from ..n.qwen.qwen_14b_chat import models as qwen_14b_chat
    from ..netman_models.qwen import nm_qwen_14b_chat
datasets = [
    *owl_cot, *owl_naive, 
    *network_cot, *network_naive, 
    *zte_cot, *zte_naive, 
    *zte_naive_ppl
]
models = [ 
    # internlm2_chat_7b,
    nm_qwen_14b_chat
]

for model in models:
    # model['path'] = model['path'].replace('/mnt/mfs/opsgpt/','/gpudata/home/cbh/opsgpt/')
    # model['tokenizer_path'] = model['tokenizer_path'].replace('/mnt/mfs/opsgpt/', '/gpudata/home/cbh/opsgpt/')
    # model['run_cfg'] = dict(num_gpus=1, num_procs=1)
    pass

zeroshot_datasets = []
fewshot_datasets = []
for dataset in datasets:
    # dataset['path'] = dataset['path'].replace('/mnt/mfs/opsgpt/evaluation','/mnt/home/opseval/evaluation/')
    dataset['sample_setting'] = dict()
    dataset['infer_cfg']['inferencer']['save_every'] = 8
    dataset['infer_cfg']['inferencer']['sc_size'] = 1
    dataset['infer_cfg']['inferencer']['max_out_len'] = 20
    dataset['eval_cfg']['sc_size'] = 1
    if 'network' in dataset['abbr']:
        dataset['sample_setting'] = dict(load_list='/mnt/tenant-home_speed/lyh/evaluation/opseval/network/network_annotated.json')
    if '3-shot' in dataset['abbr']:
        fewshot_datasets.append(dataset)
    else:
        zeroshot_datasets.append(dataset)
datasets = fewshot_datasets + zeroshot_datasets

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
