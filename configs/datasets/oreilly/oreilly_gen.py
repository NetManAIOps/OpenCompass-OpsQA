from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever, ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.utils.text_postprocessors import first_capital_postprocess_multi
from opencompass.datasets import OReillyChoiceDataset, OReillyEvaluator, oreilly_choice_postprocess

oreilly_reader_cfg = dict(
    input_columns=['topic','question','choices'],
    output_column='answer')

_end_hint = """Please answer using options A/B/C/D or their combinations. Some examples for your answers: 'A', 'B,C', 'A,C,E'.
"""

oreilly_infer_cfg = dict(
        ice_template=dict(
            type=PromptTemplate,
            template=dict(
                end=[
                    dict(role='SYSTEM', fallback_role='HUMAN', prompt=_end_hint)
                ],
                round=[
                    dict(
                        role="HUMAN",
                        prompt=
                        f"Here's a question about {{topic}}: {{question}}\n{{choices}}\nAnswer: "
                    ),
                    dict(role="BOT", prompt="{answer}\n")
                ]
            ),
            ice_token="</E>",
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )

oreilly_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    # pred_postprocessor=dict(type=first_capital_postprocess_multi))
    pred_postprocessor=dict(type='oreilly-choice'))
    

oreilly_datasets = [
    dict(
        type=OReillyChoiceDataset,
        abbr='oreilly',
        # path='/mnt/mfs/opsgpt/OpsQA/opsqa_demo/demo_oreilly.json', 
        path='/mnt/mfs/opsgpt/evaluation/ops-cert-eval/all.json', 
        reader_cfg=oreilly_reader_cfg,
        infer_cfg=oreilly_infer_cfg,
        eval_cfg=oreilly_eval_cfg)
]

del _end_hint
