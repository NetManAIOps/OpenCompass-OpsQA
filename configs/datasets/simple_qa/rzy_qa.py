from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever, ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer, SCInferencer, CoTInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator, BleuRougeEvaluator
from opencompass.utils.text_postprocessors import first_capital_postprocess_multi
from opencompass.datasets import QADataset
from mmengine.config import read_base

with read_base():
    from ...commons.cfgs import choice_qa_reader_cfg, choice_single_eval_cfg
    from ...commons.prompts import prompts
    from ...paths import ROOT_DIR
import os

rzy_qa_reader_cfg = dict(
    input_columns=['question'],
    output_column='answer',
    train_split='dev'
)

rzy_qa_eval_cfg = dict(
    evaluator=dict(type=BleuRougeEvaluator))

rzy_qa_datasets = [
    dict(
        type=QADataset,
        abbr=f'rzy_qa_{lang}-{shot_abbr}',
        path=f'{ROOT_DIR}data/opseval/rzy', 
        name=f'rzy_qa_{lang}',
        reader_cfg=rzy_qa_reader_cfg,
        infer_cfg=dict(
            ice_template=dict(
                type=PromptTemplate,
                template=dict( 
                    round=[
                        dict(
                            role="HUMAN",
                            prompt=prompt_hint
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
                            prompt=prompt_hint
                        ),
                        dict(role="BOT", prompt="{answer}\n")
                    ]
                ),
                ice_token="</E>",
            ),
            retriever=dict(type=retriever, **fixidlist),
            inferencer=dict(
                type=GenInferencer,
                save_every=20,
                generation_kwargs=dict(temperature=0),
                max_out_len = 1000,
            ),
        ),
        eval_cfg=rzy_qa_eval_cfg)
    for shot_abbr, fixidlist, shot_hint_id, retriever in zip(
            ['Zero-shot', '3-shot'],
            [dict(), dict(fix_id_list=[0,1,2])],
            [0, 1],
            [ZeroRetriever, FixKRetriever]
        )
    for lang, prompt_hint in zip(
        [
            'zh', 
            # 'en'
            ],
        [
            f"你是一名运维专家，请用一句话回答下面这个问题：\n{{question}}\n",
            # f"You are an IT operations expert, please answer the following question: \n{{question}}\n"
        ]
    )
]
