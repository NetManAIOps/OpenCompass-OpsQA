from mmengine.config import read_base
from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    # Datasets
    from ..datasets.network.network import network_naive
    # Models
    from ..local_models.qwen.qwen import qwen_1_8b_base, qwen_14b_base
    from ..netman_models.qwen import nm_qwen_14b_full_final, nm_qwen_14b_full_cp6000, nm_qwen_14b_full_cp12000, nm_qwen_14b_full_cp18000, nm_qwen_14b_full_concat, nm_qwen_14b_full_mixture, nm_qwen_14b_freeze, nm_qwen_1_8b_full_mixture, nm_qwen_1_8b_full_concat
    from ..netman_models.qwen import nm_qwen_14b_full_concat_sft_owlinstruct, nm_qwen_14b_full_mixture_sft_owlinstruct, nm_qwen_14b_full_concat_sft_owlbench, nm_qwen_14b_full_mixture_sft_owlbench, nm_qwen_14b_full_concat_sft_owlbench_instruct, nm_qwen_14b_full_mixture_sft_owlbench_instruct 
    from ..netman_models.qwen import nm_qwen_14b_freeze_3gpp_general, nm_qwen_14b_lora_3gpp_general, nm_qwen_14b_freeze_3gpp_general_sft_owlinstruct, nm_qwen_14b_lora_3gpp_general_sft_owlinstruct  

datasets = [
    *network_naive, 
]

models = [ 
    # qwen_1_8b_base,
    # qwen_14b_base,
    # nm_qwen_1_8b_full_mixture,
    # nm_qwen_1_8b_full_concat,
    # nm_qwen_14b_full_concat,
    # nm_qwen_14b_full_final,
    # nm_qwen_14b_freeze,
    # nm_qwen_14b_full_cp6000,
    # nm_qwen_14b_full_cp12000,
    # nm_qwen_14b_full_cp18000,
    # nm_qwen_14b_full_mixture,
    nm_qwen_14b_full_concat_sft_owlinstruct,
    nm_qwen_14b_full_mixture_sft_owlinstruct,
    nm_qwen_14b_full_concat_sft_owlbench,
    nm_qwen_14b_full_mixture_sft_owlbench,
    nm_qwen_14b_full_concat_sft_owlbench_instruct,
    nm_qwen_14b_full_mixture_sft_owlbench_instruct,
    # nm_qwen_14b_freeze_3gpp_general,
    # nm_qwen_14b_lora_3gpp_general,
    nm_qwen_14b_freeze_3gpp_general_sft_owlinstruct,
    nm_qwen_14b_lora_3gpp_general_sft_owlinstruct,
]

for model in models:
    model['run_cfg'] = dict(num_gpus=1, num_procs=1)
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
        max_num_workers=32,
        max_workers_per_gpu=1,   #
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
