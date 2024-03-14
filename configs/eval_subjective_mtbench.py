from mmengine.config import read_base

with read_base():
    from .datasets.subjective.multiround.mtbench_single_judge_diff_temp import subjective_datasets
    # from .datasets.subjective.multiround.mtbench_pair_judge import subjective_datasets

from opencompass.models import HuggingFaceCausalLM, HuggingFace, HuggingFaceChatGLM3
from opencompass.models.openai_api import OpenAIAllesAPIN
from opencompass.partitioners import NaivePartitioner, SizePartitioner
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.partitioners.sub_size import SubjectiveSizePartitioner
from opencompass.runners import LocalRunner
from opencompass.runners import SlurmSequentialRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.tasks.subjective_eval import SubjectiveEvalTask
from opencompass.summarizers import MTBenchSummarizer

api_meta_template = dict(
    round=[
        dict(role='SYSTEM', api_role='SYSTEM'),
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ]
)

_meta_template = dict(
    round=[
        dict(role="HUMAN", begin='\n<|im_start|>user\n', end='<|im_end|>'),
        dict(role="BOT", begin="\n<|im_start|>assistant\n", end='<|im_end|>', generate=True),
    ],
)
# -------------Inference Stage ----------------------------------------
# For subjective evaluation, we often set do sample for models
models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='qwen-7b-chat-hf',
        path="Qwen/Qwen-7B-Chat",
        tokenizer_path='Qwen/Qwen-7B-Chat',
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
            use_fast=False,
        ),
        pad_token_id=151643,
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        meta_template=_meta_template,
        run_cfg=dict(num_gpus=1, num_procs=1),
        end_str='<|im_end|>',
    )
]

datasets = [*subjective_datasets]

infer = dict(
    partitioner=dict(type=SizePartitioner, strategy='split', max_task_size=10000),
    runner=dict(
        type=SlurmSequentialRunner,
        partition='llm_dev2',
        quotatype='auto',
        max_num_workers=256,
        task=dict(type=OpenICLInferTask),
    ),
)

# -------------Evalation Stage ----------------------------------------

## ------------- JudgeLLM Configuration
judge_model = dict(
    abbr='GPT4-Turbo',
    type=OpenAIAllesAPIN,
    path='gpt-4-0613', # To compare with the official leaderboard, please use gpt4-0613
    key='xxxx',  # The key will be obtained from $OPENAI_API_KEY, but you can write down your key here as well
    url='xxxx',
    meta_template=api_meta_template,
    query_per_second=16,
    max_out_len=2048,
    max_seq_len=2048,
    batch_size=8,
    temperature=0,
)
## ------------- Evaluation Configuration
# ## pair evaluation
# eval = dict(
#     partitioner=dict(
#         type=SubjectiveSizePartitioner, max_task_size=100, mode='m2n', base_models=[gpt4], compare_models=models
#     ),
#     runner=dict(type=LocalRunner, max_num_workers=32, task=dict(type=SubjectiveEvalTask, judge_cfg=judge_model)),
# )

# summarizer = dict(type=MTBenchSummarizer, judge_type='pair')


## single evaluation
eval = dict(
    partitioner=dict(type=SubjectiveSizePartitioner, strategy='split', max_task_size=10000, mode='singlescore', models=models),
    runner=dict(type=LocalRunner, max_num_workers=32, task=dict(type=SubjectiveEvalTask, judge_cfg=judge_model)),
)

summarizer = dict(type=MTBenchSummarizer, judge_type='single')

work_dir = 'outputs/mtbench/'
