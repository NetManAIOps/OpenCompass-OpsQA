from mmengine.config import read_base
from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import HuggingFace, HuggingFaceCausalLM, TurboMindModel, LmdeployPytorchModel

with read_base():
    # Datasets
    from ..datasets.opseval.datasets import ceval_mc_ppl, network_mc_ppl, zte_mc_ppl, owl_mc_ppl, ceval_mc_gen, network_mc_gen, zte_mc_gen, owl_mc_gen, owl_qa_gen, owl_qa_ppl, rzy_qa_gen, rzy_qa_ppl, oracle_mc_gen, oracle_mc_ppl, company_mc_gen, company_mc_ppl
    from ..datasets.simple_qa.owl_qa import owl_qa_datasets
    from ..datasets.ppl_qa.owl_qa import owl_ppl_qa_datasets
    from ..datasets.simple_qa.rzy_qa import rzy_qa_datasets
    from ..datasets.ppl_qa.rzy_qa import rzy_ppl_qa_datasets
    from ..datasets.zte.zte import zte_naive
datasets = [*ceval_mc_ppl,*network_mc_ppl,*zte_mc_ppl,*owl_mc_ppl,*oracle_mc_ppl,*company_mc_ppl,*ceval_mc_gen,*network_mc_gen,*zte_mc_gen,*owl_mc_gen,*oracle_mc_gen,*company_mc_gen,*owl_qa_gen,*owl_qa_ppl,*rzy_qa_gen,*rzy_qa_ppl]
model_name = 'qwen-14b-chat'
model_abbr = 'nm_qwen_14b_chat'
model_path = '/mnt/home/opsfm-xz/models/Qwen/Qwen-14B-Chat'

if model_name is None:
    raise NotImplementedError("Model is none!")
models = [dict(
            type=TurboMindModel,
            abbr=model_abbr,
            path=model_path,
            engine_config=dict(session_len=2048,
                           max_batch_size=8),
            gen_config=dict(top_k=1, top_p=0.8,
                        max_new_tokens=100, stop_words=[151643, 151645]),
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
            end_str='<|im_end|>'
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
    # dataset['infer_cfg']['inferencer']['generation_kwargs'] = {'stopping_criteria': ['<|im_end|>', '<|endoftext|>']}
    if 'qa' in dataset['abbr'].replace('-', '_').split('_'):
        dataset['infer_cfg']['inferencer']['max_out_len'] = 50
    if 'en' in dataset['abbr'].replace('-', '_').split('_'):
        continue
    if 'sc+cot' in dataset['abbr']:
        continue
    if 'mc' in dataset['abbr'].replace('-', '_').split('_'):
        dataset['sample_setting']= dict(sample_size=500)
    if 'owl_qa' in dataset['abbr']:
        dataset['sample_setting']= dict(sample_size=500)
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
