from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever, ZeroRetriever
from opencompass.openicl.icl_inferencer import SCInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.utils.text_postprocessors import first_capital_postprocess_multi
from opencompass.datasets import OReillyChoiceDataset, OReillyEvaluator, oreilly_choice_postprocess

SAMPLE_SIZE = 3

oreilly_reader_cfg = dict(
    input_columns=['topic','question','choices','qtype','hint'],
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
                    f"Here is a {{qtype}} question about {{topic}}.\n\n{{question}}\n{{choices}}\n\n{{hint}}\n\nAnswer:\n"
                ),
                dict(role="BOT", prompt="{answer}")
            ]
        ),
        ice_token="</E>",
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(
        type=SCInferencer,
        # generation_kwargs=dict(do_sample=True, temperature=0.7, top_k=40),
        generation_kwargs=dict(temperature=0.7),
        infer_type='SC',
        sc_size = SAMPLE_SIZE
    ),
)

oreilly_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    sc_size=SAMPLE_SIZE,
    # pred_postprocessor=dict(type=first_capital_postprocess_multi))
    pred_postprocessor=dict(type='oreilly-choice')
)
    

oreilly_datasets = [
    dict(
        type=OReillyChoiceDataset,
        abbr='oreilly',
        # path='/mnt/mfs/opsgpt/OpsQA/opsqa_demo/demo_oreilly.json', 
        path='/mnt/mfs/opsgpt/evaluation/ops-cert-eval', 
        filename='all.json',
        # sample_setting=dict(
        #     seed=233,
        #     sample_size=50,
        # ),
        reader_cfg=oreilly_reader_cfg,
        infer_cfg=oreilly_infer_cfg,
        eval_cfg=oreilly_eval_cfg)
]

