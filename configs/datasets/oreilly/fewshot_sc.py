from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever, ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer, SCInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.utils.text_postprocessors import first_capital_postprocess_multi
from opencompass.datasets import OReillyChoiceDataset, OReillyEvaluator, oreilly_choice_postprocess, OReillyDataset

SAMPLE_SIZE = 5

oreilly_reader_cfg = dict(
    input_columns=['topic','question','choices','qtype'],
    output_column='answer',
    train_split='dev'
)

oreilly_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    sc_size=SAMPLE_SIZE, 
    pred_postprocessor=dict(type='oreilly-choice'))
    

oreilly_datasets = [
    dict(
        type=OReillyDataset,
        abbr=f'oreilly-3shot-sc-{qtype_abbr}',
        path='/mnt/mfs/opsgpt/evaluation/ops-cert-eval/v3', 
        name=f'{qtype_abbr}',
        qtype=qtype_id,
        # sample_setting=dict(
        #     sample_size=1
        # ), 
        reader_cfg=oreilly_reader_cfg,
        infer_cfg=dict(
            ice_template=dict(
                type=PromptTemplate,
                template=dict(
                    begin="</E>", 
                    round=[
                        dict(
                            role="HUMAN",
                            prompt=f"Here is a {{qtype}} question about {{topic}}.\n{{question}}\n{{choices}}\n{hint}Answer:\n"
                        ),
                        dict(role="BOT", prompt="{answer}\n")
                    ]
                ),
                ice_token="</E>",
            ),
            retriever=dict(type=FixKRetriever),
            inferencer=dict(
                type=SCInferencer,
                generation_kwargs=dict(temperature=0.7),
                infer_type='SC',
                sc_size = SAMPLE_SIZE, 
                save_every=200,
                fix_id_list=[0,1,2]
            ),
        ),
        eval_cfg=oreilly_eval_cfg)
    for qtype_abbr, qtype_id, hint in zip(
        ['single', 'multiple'],
        [0, 1],
        ['', 'Some examples for your answer: A,B | B,C,D | A,E\n']
    )
]

