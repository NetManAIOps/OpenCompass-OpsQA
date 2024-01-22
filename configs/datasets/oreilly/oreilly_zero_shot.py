from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever, ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.utils.text_postprocessors import first_capital_postprocess_multi
from opencompass.datasets import OReillyChoiceDataset, OReillyEvaluator, oreilly_choice_postprocess

oreilly_reader_cfg = dict(
    input_columns=['topic','question','choices','qtype'],
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
                    f"Here is a {{qtype}} question about {{topic}}.\n{{question}}\n{{choices}}\nAnswer:\n"
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
        abbr='oreilly-zero-shot',
        # path='/home/v-xll22/OpsGPT', 
        # filename='unused.json',
        path='/mnt/mfs/opsgpt/evaluation/ops-cert-eval', 
        filename='all.json',
        sample_setting=dict(
            seed=233,
            sample_size=500,
        ),
        reader_cfg=oreilly_reader_cfg,
        infer_cfg=oreilly_infer_cfg,
        eval_cfg=oreilly_eval_cfg)
]

