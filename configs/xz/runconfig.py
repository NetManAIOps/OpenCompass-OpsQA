from mmengine.config import read_base
from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import HuggingFace, HuggingFaceCausalLM

with read_base():
    # Datasets
    from ..datasets.company.company import company_cot, company_naive
    from ..datasets.network.network import network_cot, network_naive, network_ppl
    from ..datasets.zte.zte import zte_cot, zte_naive, zte_ppl
    from ..datasets.oracle.oracle import oracle_cot, oracle_naive
    from ..datasets.owl_mc.owl_mc import owl_cot, owl_naive, owl_ppl
    # from ..datasets.zte.zte_ppl import zte_naive_ppl
    from ..datasets.simple_qa.owl_qa import owl_qa_datasets
    from ..datasets.ppl_qa.owl_qa import owl_ppl_qa_datasets
    from ..datasets.simple_qa.rzy_qa import rzy_qa_datasets
    from ..datasets.ppl_qa.rzy_qa import rzy_ppl_qa_datasets
    from ..datasets.opseval_ceval.ceval_ppl import ceval_naive_ppl
    from ..datasets.opseval_ceval.ceval_naive import ceval_naive
    #/mnt/tenant-home_speed/lyh/OpenCompass-OpsQA-lyh/models/xll_models/Qwen-1_8B/mixture/mix5_1-sft_owl-model
    # Models
    # from ..netman_models.qwen import nm_qwen_14b_3gpp_mix_sft_owlb as zzr1
    # from ..netman_models.qwen import nm_qwen_14b_ptlora_book as zzr1
    # from ..netman_models.qwen import *
    # from ..netman_models.qwen import nm_qwen_1_8b_books_all_lora as zzr1
datasets = [*zte_ppl,*network_ppl,*owl_ppl,*owl_qa_datasets,*owl_ppl_qa_datasets,*ceval_naive_ppl,*zte_naive,*owl_naive,*network_naive,*ceval_naive]
# datasets = [
#     *zte_ppl, *network_ppl, *owl_ppl,
#     *owl_qa_datasets,
#     *owl_ppl_qa_datasets,
#     *ceval_naive_ppl,
#     *zte_cot, *zte_naive,     
#     *owl_cot, *owl_naive, 
#     *network_cot, *network_naive, 
# ]

model_name = 'qwen-14b-full-dsir_10000-mmlu-sft-20epochs'
model_abbr = 'nm_qwen_14b_qwen_14b_full_dsir_10000_mmlu_sft_20epochs'
model_path = '/mnt/tenant-home_speed/xz/sft_checkpoint/qwen-14b-full-dsir_10000-mmlu-sft-20epochs'

if model_name is None:
    raise NotImplementedError("Model is none!")
models = [dict(
        type=HuggingFaceCausalLM,
        abbr=model_abbr,
        path=model_path,
        tokenizer_path=model_path,
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              trust_remote_code=True,
                              use_fast=False,),
        max_out_len=400,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )]

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
