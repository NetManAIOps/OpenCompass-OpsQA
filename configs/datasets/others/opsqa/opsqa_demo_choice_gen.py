from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever, ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.utils.text_postprocessors import first_capital_postprocess
from opencompass.datasets import OpsQAChoiceDemoDataset

opsqa_reader_cfg = dict(
    input_columns=['input','A','B','C','D'],
    output_column='target')

_hint = '这里有一个关于运维的选择题，用A, B, C或D来回答。'

opsqa_infer_cfg = dict(
        ice_template=dict(
            type=PromptTemplate,
            template=dict(round=[
                dict(
                    role="HUMAN",
                    prompt=
                    f"{_hint}\n问题: {{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\n答案: "
                ),
                dict(role="BOT", prompt="{target}\n")
            ]),
            ice_token="</E>",
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer, labels=['A', 'B', 'C', 'D']),
    )

opsqa_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_postprocessor=dict(type=first_capital_postprocess))

opsqa_datasets = [
    dict(
        type=OpsQAChoiceDemoDataset,
        abbr='opsqa_demo_choice_gen',
        path='/mnt/mfs/opsgpt/OpsQA/opsqa_demo/demo_csv.csv',       # for OpsQADataset load function
        reader_cfg=opsqa_reader_cfg,
        infer_cfg=opsqa_infer_cfg,
        eval_cfg=opsqa_eval_cfg)
]

del _hint
