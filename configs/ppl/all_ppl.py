from mmengine.config import read_base
from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    # Datasets
    from ..datasets.zte.zte_ppl import zte_naive
    from ..datasets.ppl_qa.owl_qa import owl_ppl_qa_datasets
    from ..datasets.opseval_ceval.ceval_ppl import ceval_naive_ppl
    # Models
    from ..local_models.qwen.qwen import qwen1_5_14b_base
    from ..local_models.qwen.qwen import qwen_14b_base
    from ..netman_models.qwen import netman_qwen_models, nm_qwen_14b_lora_3gpp_general, nm_qwen_1_8b_mix5_1
datasets = [
    # *zte_naive, 
    *owl_ppl_qa_datasets,
    # *ceval_naive_ppl,
]

models = [
    # qwen1_5_14b_base,
    # qwen_14b_base,
    # *netman_qwen_models
    # nm_qwen_14b_lora_3gpp_general,
    nm_qwen_1_8b_mix5_1
]

for model in models:
    pass

for dataset in datasets:
    dataset['sample_setting'] = dict()
    dataset['infer_cfg']['inferencer']['save_every'] = 8
    dataset['infer_cfg']['inferencer']['sc_size'] = 1
    dataset['infer_cfg']['inferencer']['max_out_len'] = 20
    dataset['eval_cfg']['sc_size'] = 1
    if 'network' in dataset['abbr']:
        dataset['sample_setting'] = dict(load_list='/mnt/tenant-home_speed/lyh/evaluation/opseval/network/network_annotated.json')

infer = dict(
    partitioner=dict(
        # type=SizePartitioner,
        # max_task_size=100,
        # gen_task_coef=1,
        model_first=True,
        type=NaivePartitioner
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=16,
        max_workers_per_gpu=2,
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
