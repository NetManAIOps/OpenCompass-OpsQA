from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.utils.text_postprocessors import first_capital_postprocess
from opencompass.datasets import OpsQADataset

opsqa_reader_cfg = dict(
    input_columns=['input','A','B','C','D'],
    output_column='target')

opsqa_prompt_template = dict(
    type='PromptTemplate',
    template=None,
    ice_token='</E>')

opsqa_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: '
            ),
            dict(role='BOT', prompt='{target}\n')
        ])),
    prompt_template=opsqa_prompt_template,
    retriever=dict(type=FixKRetriever),
    inferencer=dict(type=GenInferencer, fix_id_list=[0, 1, 2, 3, 4]))

opsqa_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_postprocessor=dict(type=first_capital_postprocess))

triviaqa_datasets = [
    dict(
        type=OpsQADataset,
        abbr='opsqa',
        path='./data/OpsQA/',       # for OpsQADataset load function
        reader_cfg=opsqa_reader_cfg,
        infer_cfg=opsqa_infer_cfg,
        eval_cfg=opsqa_eval_cfg)
]
