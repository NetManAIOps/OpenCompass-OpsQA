
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever, ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer, SCInferencer, CoTInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator, BleuRougeEvaluator
from opencompass.utils.text_postprocessors import first_capital_postprocess_multi
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.datasets import OpsEvalQADataset
from mmengine import read_base
with read_base():
    from ...commons.cfgs import mc_abcd_reader_cfg
    from ...commons.prompts import prompts
    from ...commons.inferencers import get_gen_inferencer, get_cot_inferencer
    from ...commons.templates import qa_gen_ice_template, qa_gen_prompt_template
    from ...paths import ROOT_DIR

SAMPLE_SIZE = 3

# TODO: 写一个装饰器函数，可以完成shot, qtype, lang的循环生成


def get_qa_gen_datasets(dataset_name, path, langs=['zh'], qtypes=None):
    naive_gen_datasets = [
        dict(
            type=OpsEvalQADataset,
            abbr=f'{dataset_name}-qa-{shot_abbr}-{lang}-sc-gen',
            path=path, 
            name=f'{dataset_name}_qa_{lang}',
            reader_cfg=mc_abcd_reader_cfg,
            infer_cfg=dict(
                ice_template=qa_gen_ice_template(prompt_hint, answer_hint),
                prompt_template=qa_gen_prompt_template(prompt_hint, answer_hint),
                retriever=retriever_dict,
                inferencer=get_gen_inferencer(sc=False),
            ),
            eval_cfg=dict(evaluator=dict(type=BleuRougeEvaluator))
            )
            for shot_abbr, shot_hint_id, retriever_dict in zip(
                ['Zero-shot', '3-shot'],
                [0, 1],
                [dict(type=ZeroRetriever), dict(type=FixKRetriever, fix_id_list=[0,1,2])]
            )
            for lang, prompt_hint, answer_hint in zip(
                ['zh', 'en'],
                [
                    f"你是一名运维专家，请回答下面这个问题：\n",
                    f"You are an IT operations expert, please answer the following question: \n"
                ],
                [
                    "答案：",
                    "Answer:"
                ]
            )
    ]
    datasets = naive_gen_datasets
    selected = []
    for lang in langs:
        selected.extend([d for d in datasets if f'{lang}' in d['abbr']])
    return selected


zjyd_qa_gen_datasets = get_qa_gen_datasets('zjyd', f'{ROOT_DIR}data/opseval/zjyd/')