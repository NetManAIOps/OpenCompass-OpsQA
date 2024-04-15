from mmengine.config import read_base
from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import HuggingFace, HuggingFaceCausalLM, TurboMindModel, LmdeployPytorchModel, VLLM

with read_base():
    # Datasets
    from ..datasets.opseval.datasets import ceval_mc_ppl, network_mc_ppl, zte_mc_ppl, owl_mc_ppl, ceval_mc_gen, network_mc_gen, zte_mc_gen, owl_mc_gen, owl_qa_gen, owl_qa_ppl, rzy_qa_gen, rzy_qa_ppl, zedx_qa_gen, zedx_qa_ppl, oracle_mc_gen, oracle_mc_ppl, company_mc_gen, company_mc_ppl
    from ..datasets.simple_qa.owl_qa import owl_qa_datasets
    from ..datasets.ppl_qa.owl_qa import owl_ppl_qa_datasets
    from ..datasets.simple_qa.rzy_qa import rzy_qa_datasets
    from ..datasets.ppl_qa.rzy_qa import rzy_ppl_qa_datasets
    from ..datasets.zte.zte import zte_naive


# models = [$MODEL_TEMPLATE$]

# for model in models:
#     # model['path'] = model['path'].replace('/mnt/mfs/opsgpt/','/gpudata/home/cbh/opsgpt/')
#     # model['tokenizer_path'] = model['tokenizer_path'].replace('/mnt/mfs/opsgpt/', '/gpudata/home/cbh/opsgpt/')
#     # model['run_cfg'] = dict(num_gpus=1, num_procs=1)
#     pass

model_dataset_combinations = [{
    'models': [dict(
            type=VLLM,
            abbr='nm_qwen1.5_32b_zedx_full_2000step_sft_2000step',
            path='/mnt/tenant-home_speed/xz/sft_checkpoint/qwen1.5-32b-zedx-full-2000step-sft-2000step/merged_model',
            max_out_len=400,
            max_seq_len=2048,
            batch_size=8,
            run_cfg=dict(num_gpus=1, num_procs=1),
            meta_template=dict(
        round=[
            dict(role="HUMAN", begin='<|im_start|>user\n', end='<|im_end|>'),
            dict(role="BOT", begin="<|im_start|>assistant\n", end='<|im_end|>', generate=True),
        ],
        reserved_roles=[dict(role='SYSTEM', begin="<|im_start|>system\nYou are a helpful assistant.", end="<|im_end|>"),]
    ),
            model_kwargs=dict(trust_remote_code=True, max_model_len=2000, tensor_parallel_size=1, gpu_memory_utilization=0.95),
            generation_kwargs=dict(stop_token_ids=[151643, 151645]),
            mode='none',
            use_fastchat_template=False
        )],
    'datasets': []
}, {
    'models': [dict(
            type=HuggingFaceCausalLM,
            abbr='nm_qwen1.5_32b_zedx_full_2000step_sft_2000step',
            path='/mnt/tenant-home_speed/xz/sft_checkpoint/qwen1.5-32b-zedx-full-2000step-sft-2000step/merged_model',
            tokenizer_path='/mnt/tenant-home_speed/xz/sft_checkpoint/qwen1.5-32b-zedx-full-2000step-sft-2000step/merged_model',
            tokenizer_kwargs=dict(padding_side='left',
                                truncation_side='left',
                                trust_remote_code=True,
                                use_fast=False,),
            max_out_len=400,
            max_seq_len=2048,
            batch_size=8,
            model_kwargs=dict(device_map='auto', trust_remote_code=True),
            run_cfg=dict(num_gpus=1, num_procs=1),
            meta_template=dict(
        round=[
            dict(role="HUMAN", begin='<|im_start|>user\n', end='<|im_end|>'),
            dict(role="BOT", begin="<|im_start|>assistant\n", end='<|im_end|>', generate=True),
        ],
        reserved_roles=[dict(role='SYSTEM', begin="<|im_start|>system\nYou are a helpful assistant.", end="<|im_end|>"),]
    ),
            generation_kwargs=dict(eos_token_id=[151643, 151645])
        )],
    'datasets': []
}]

zeroshot_datasets = []
fewshot_datasets = []

for dataset in [*ceval_mc_ppl,*network_mc_ppl,*zte_mc_ppl,*owl_mc_ppl,*oracle_mc_ppl,*company_mc_ppl,*ceval_mc_gen,*network_mc_gen,*zte_mc_gen,*owl_mc_gen,*oracle_mc_gen,*company_mc_gen,*zedx_qa_gen,*zedx_qa_ppl]:
    # dataset['path'] = dataset['path'].replace('/mnt/mfs/opsgpt/evaluation','/mnt/home/opseval/evaluation/')
    dataset['sample_setting'] = dict()
    dataset['infer_cfg']['inferencer']['save_every'] = 8
    dataset['infer_cfg']['inferencer']['sc_size'] = 1
    dataset['infer_cfg']['inferencer']['max_out_len'] = 20
    # dataset['infer_cfg']['inferencer']['generation_kwargs'] = {'stopping_criteria': ['<|im_end|>', '<|endoftext|>']}
    if 'qa' in dataset['abbr'].replace('-', '_').split('_'):
        if 'zedx' in dataset['abbr']:
            dataset['infer_cfg']['inferencer']['max_out_len'] = 100
        else:
            dataset['infer_cfg']['inferencer']['max_out_len'] = 50
    if 'en' in dataset['abbr'].replace('-', '_').split('_'):
        continue
    if 'sc+cot' in dataset['abbr']:
        continue
    if 'mc' in dataset['abbr'].replace('-', '_').split('_'):
        dataset['sample_setting']= dict(sample_size=500)
    if 'qa' in dataset['abbr'].replace('-', '_').split('_') and 'ppl' not in dataset['abbr']:
        dataset['sample_setting']= dict(sample_size=500)
    dataset['eval_cfg']['sc_size'] = 1
    if 'network' in dataset['abbr']:
        dataset['sample_setting'] = dict(load_list='/mnt/tenant-home_speed/lyh/evaluation/opseval/network/network_annotated.json')
    
    if 'ppl' in dataset['abbr']:
        model_dataset_combinations[1]['datasets'].append(dataset)
    else:
        model_dataset_combinations[0]['datasets'].append(dataset)
#     if '3-shot' in dataset['abbr']:
#         fewshot_datasets.append(dataset)
#     else:
#         zeroshot_datasets.append(dataset)
# datasets = fewshot_datasets + zeroshot_datasets

models = [*model_dataset_combinations[0]['models'], *model_dataset_combinations[1]['models']]
datasets = [*model_dataset_combinations[0]['datasets'], *model_dataset_combinations[1]['datasets']]

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
        max_num_workers=8,
        task=dict(type=OpenICLEvalTask)),
)
