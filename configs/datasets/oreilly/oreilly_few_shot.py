from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever, ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.utils.text_postprocessors import first_capital_postprocess_multi
from opencompass.datasets import OReillyChoiceDataset, OReillyEvaluator, oreilly_choice_postprocess

oreilly_reader_cfg = dict(
    input_columns=['topic','question','choices','qtype','examples'],
    output_column='answer'
)

oreilly_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role="HUMAN",
                    prompt=
                    f"{{examples}}Answer the following {{qtype}} question about {{topic}}.\nQuestion: {{question}}\n{{choices}}\nAnswer:"
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

oreilly_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    # pred_postprocessor=dict(type=first_capital_postprocess_multi))
    pred_postprocessor=dict(type='oreilly-choice'))
    

oreilly_datasets = [
    dict(
        type=OReillyChoiceDataset,
        abbr='oreilly-few-shot',
        path='/home/v-xll22/OpsGPT',
        # path='/mnt/mfs/opsgpt/evaluation/ops-cert-eval',
        filename='fewshot.json',
        few_shot=True,
        reader_cfg=oreilly_reader_cfg,
        infer_cfg=oreilly_infer_cfg,
        eval_cfg=oreilly_eval_cfg)
]

