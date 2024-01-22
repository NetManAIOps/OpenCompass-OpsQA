from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever, ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer, SCInferencer, CoTInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.utils.text_postprocessors import first_capital_postprocess_multi
from opencompass.datasets import QAZeroshotDataset
import os

zte_reader_cfg = dict(
    input_columns=['question'],
    output_column='answer',
    train_split='dev'
)

zte_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator))

zte_datasets = [
    dict(
        type=QAZeroshotDataset,
        abbr=f'zte_qa_{filename}-Zeroshot',
        path=f'/mnt/mfs/opsgpt/evaluation/opseval/zte/qa/', 
        name=f'{filename}',
        reader_cfg=zte_reader_cfg,
        infer_cfg=dict(
            ice_template=dict(
                type=PromptTemplate,
                template=dict( 
                    round=[
                        dict(
                            role="HUMAN",
                            prompt=f"你是一名运维专家，请回答下面这个问题：\n{{question}}\n"
                        ),
                        dict(role="BOT", prompt=f"{{answer}}\n"),
                    ]
                ),
            ),
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    begin="</E>", 
                    round=[
                        dict(
                            role="HUMAN",
                            prompt=f"你是一名运维专家，请回答下面这个问题：\n{{question}}\n"
                        ),
                        dict(role="BOT", prompt="{answer}\n")
                    ]
                ),
                ice_token="</E>",
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(
                type=GenInferencer,
                save_every=20,
                generation_kwargs=dict(temperature=0),
                max_out_len = 1000,
            ),
        ),
        eval_cfg=zte_eval_cfg)
    for filename in [
        'EMSPlus',
        'NetCloud',
        'USPP',
        'CG',
        'XGW',
        'VNFM',
        'PCF',
        'GSO',
        'EMS',
        'WYY',
        'Openstack',
        'CNIA',
        'Director',
        'DSC',
        'CCNuMAC'
    ]
]

