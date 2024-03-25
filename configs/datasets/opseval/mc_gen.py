
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever, ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer, SCInferencer, CoTInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator, OpsEvalGenMCEvaluator
from opencompass.utils.text_postprocessors import first_capital_postprocess_multi
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.datasets import OpsEvalMCDataset
from mmengine import read_base
with read_base():
    from ...commons.cfgs import mc_abcd_reader_cfg
    from ...commons.prompts import prompts
    from ...commons.inferencers import get_gen_inferencer, get_cot_inferencer
    from ...commons.templates import mc_abcd_gen_ice_template, mc_abcd_gen_prompt_template, mc_abcd_cot_ice_template, mc_abcd_cot_prompt_template
    from ...paths import ROOT_DIR

SAMPLE_SIZE = 3

# TODO: 写一个装饰器函数，可以完成shot, qtype, lang的循环生成


def get_mc_gen_datasets(dataset_name, path, langs=['zh'], qtypes=['single']):
    naive_gen_datasets = [
        dict(
            type=OpsEvalMCDataset,
            abbr=f'{dataset_name}-mc-{shot_abbr}-{lang}-{qtype}-sc-gen',
            path=path, 
            name=f'{dataset_name}_mc_{lang}_{qtype}',
            reader_cfg=mc_abcd_reader_cfg,
            infer_cfg=dict(
                ice_template=mc_abcd_gen_ice_template(prompt_hint, answer_hint),
                prompt_template=mc_abcd_gen_prompt_template(prompt_hint, answer_hint),
                retriever=retriever_dict,
                inferencer=get_gen_inferencer(sc_size=SAMPLE_SIZE),
            ),
            eval_cfg=dict(evaluator=dict(type=OpsEvalGenMCEvaluator))
            )
            for shot_abbr, shot_hint_id, retriever_dict in zip(
                ['Zero-shot', '3-shot'],
                [0, 1],
                [dict(type=ZeroRetriever), dict(type=FixKRetriever, fix_id_list=[0,1,2])]
            )
            for qtype, qtype_hint_id in zip(
                qtypes,
                [0 if qt == 'single' else 1 for qt in qtypes]
            )
            for lang, prompt_hint, answer_hint in zip(
                # ['zh'],
                langs,
                [
                    f"{prompts[shot_hint_id][qtype_hint_id][1 if l == 'zh' else 0]}"
                    for l in langs
                ],
                [
                    "请直接给出正确选项：" if l == 'zh' else "Please provide the correct answer option directly: " for l in langs
                ]
            )
    ]

    cot_gen_datasets = [
        dict(
            type=OpsEvalMCDataset,
            abbr=f'{dataset_name}-mc-{shot_abbr}-{lang}-{qtype}-sc+cot-gen',
            path=path, 
            name=f'{dataset_name}_mc_{lang}_{qtype}',
            reader_cfg=mc_abcd_reader_cfg,
            infer_cfg=dict(
                ice_template=mc_abcd_cot_ice_template(prompt_hint, cot_think_hint, cot_conclude_hint),
                prompt_template=mc_abcd_cot_prompt_template(prompt_hint, cot_think_hint),
                retriever=retriever_dict,
                inferencer=get_cot_inferencer(sc_size=SAMPLE_SIZE, cot_prompts=cot_conclude_hint),
            ),
            eval_cfg=dict(evaluator=dict(type=OpsEvalGenMCEvaluator)))
        for shot_abbr, shot_hint_id, retriever_dict in zip(
            ['Zero-shot', '3-shot'],
            [0, 1],
            [dict(type=ZeroRetriever), dict(type=FixKRetriever, fix_id_list=[0,1,2])]
        )
        for qtype, qtype_hint_id in zip(
            # ['single', 'multiple'],
            qtypes,
            [0 if qt == 'single' else 1 for qt in qtypes]
        )
        for lang, prompt_hint, cot_think_hint, cot_conclude_hint in zip(
            # ['en', 'zh'],
            langs,
            [
                "Here is a multiple-choice question.\n" if l == 'en' 
                else "以下是一个选择题。\n" for l in langs
            ],
            [
                f"Let's think step by step.\n" if l == 'en'
                else f"让我们逐个选项分析：\n" for l in langs
            ],
            [
                f'{prompts[shot_hint_id][qtype_hint_id][0]}Therefore the answer is: \n' if l == 'en'
                else f'{prompts[shot_hint_id][qtype_hint_id][1]}因此答案是：\n' for l in langs
            ]
        )
    ]

    datasets = naive_gen_datasets + cot_gen_datasets
    selected = []
    for lang in langs:
        for qtype in qtypes:
            selected.extend([d for d in datasets if f'{lang}-{qtype}' in d['abbr']])
    return selected


zjyd_mc_gen_datasets = get_mc_gen_datasets('zjyd', f'{ROOT_DIR}data/opseval/zjyd/')