
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever, ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer, SCInferencer, CoTInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.utils.text_postprocessors import first_capital_postprocess_multi
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.datasets import OpsEvalMCDataset
from mmengine import read_base
with read_base():
    from ...commons.cfgs import choice_qa_reader_cfg, choice_single_eval_cfg
    from ...commons.prompts import prompts
    from ...commons.inferencers import get_ppl_inferencer
    from ...commons.templates import mc_abcd_ppl_ice_template, mc_abcd_ppl_prompt_template
    from ...paths import ROOT_DIR

SAMPLE_SIZE = 3

def get_mc_ppl_datasets(dataset_name, path, langs=['zh'], qtypes=['single']):
    naive_ppl_datasets = [
        dict(
            type=OpsEvalMCDataset,
            abbr=f'{dataset_name}-mc-{shot_abbr}-{lang}-{qtype}-sc-ppl',
            path=path, 
            name=f'{dataset_name}_mc_{lang}_{qtype}',
            reader_cfg=dict(
                input_columns=['question','A', 'B', 'C', 'D'],
                output_column='answer',
                train_split='dev'
            ),
            infer_cfg=dict(
                ice_template=mc_abcd_ppl_ice_template(prompt_hint, answer_hint),
                prompt_template=mc_abcd_ppl_prompt_template(prompt_hint, answer_hint),
                retriever=retriever_dict,
                inferencer=get_ppl_inferencer(),
            ),
            eval_cfg=dict(evaluator=dict(type=AccEvaluator)))
            for shot_abbr, shot_hint_id, retriever_dict in zip(
                ['Zero-shot', '3-shot'],
                [0, 1],
                [dict(type=ZeroRetriever), dict(type=FixKRetriever, fix_id_list=[0,1,2])]
            )
            for qtype, qtype_hint_id in zip(
                ['single'],
                [0]
            )
            for lang, prompt_hint, answer_hint in zip(
                ['zh'],
                [
                    f"{prompts[shot_hint_id][qtype_hint_id][1]}"
                ],
                [
                    '答案：',
                    'Answer:'
                ]
            )
    ]

    datasets = naive_ppl_datasets
    selected = []
    for lang in langs:
        for qtype in qtypes:
            selected.extend([d for d in datasets if f'{lang}-{qtype}' in d['abbr']])
    return selected

zjyd_mc_ppl_datasets = get_mc_ppl_datasets('zjyd', f'{ROOT_DIR}data/opseval/zjyd/')