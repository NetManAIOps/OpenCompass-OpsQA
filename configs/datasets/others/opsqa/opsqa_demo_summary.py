from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever, ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.utils.text_postprocessors import first_capital_postprocess
from opencompass.datasets import OpsQASummaryDemoDataset, OpsQAEvaluator

opsqa_reader_cfg = dict(
    input_columns=['context', 'hint'],
    output_column='answer')

opsqa_infer_cfg = dict(
        ice_template=dict(
            type=PromptTemplate,
            template=dict(round=[
                dict(
                    role="HUMAN",
                    prompt=
                    f"{{context}}\n{{hint}}"
                ),
                dict(role="BOT", prompt="{answer}\n")
            ]),
            ice_token="</E>",
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )

opsqa_eval_cfg = dict(
    evaluator=dict(type=OpsQAEvaluator))

opsqa_datasets = [
    dict(
        type=OpsQASummaryDemoDataset,
        abbr='opsqa_demo_summary',
        path='/mnt/mfs/opsgpt/OpsQA/opsqa_demo/demo_abstract.json',       # for OpsQADataset load function
        reader_cfg=opsqa_reader_cfg,
        infer_cfg=opsqa_infer_cfg,
        eval_cfg=opsqa_eval_cfg)
]
