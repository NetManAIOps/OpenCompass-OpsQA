from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import OpsQADataset, OpsQAEvaluator

opsqa_reader_cfg = dict(
    input_columns=['question','topic','task'],
    output_column='answer'
)

opsqa_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role="HUMAN",
                    prompt=
                    f"Answer the {{task}} question about {{topic}}.\n{{question}}\nAnswer:\n"
                ),
                dict(role="BOT", prompt="{answer}")
            ]
        ),
        ice_token="</E>",
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(
        type=GenInferencer,
        save_every=200
    ),
)

opsqa_eval_cfg = dict(
    evaluator=dict(type=OpsQAEvaluator)
)

opsqa_datasets = [
    dict(
        type=OpsQADataset,
        abbr='OpsQA',
        path='/gpudata/home/cbh/opsgpt/evaluation/ops-cert-eval/opsqa.json',
        reader_cfg=opsqa_reader_cfg,
        infer_cfg=opsqa_infer_cfg,
        eval_cfg=opsqa_eval_cfg)
]

