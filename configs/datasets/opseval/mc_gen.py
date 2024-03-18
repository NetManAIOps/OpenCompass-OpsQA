
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
                retriever=dict(type=retriever, **fixidlist),
                inferencer=get_gen_inferencer(sc_size=SAMPLE_SIZE, fixidlist=fixidlist),
            ),
            eval_cfg=dict(evaluator=dict(type=OpsEvalGenMCEvaluator))
            )
            for shot_abbr, fixidlist, shot_hint_id, retriever in zip(
                ['Zero-shot', '3-shot'],
                [dict(), dict(fix_id_list=[0, 1, 2])],
                [0, 1],
                [ZeroRetriever, FixKRetriever]
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
                    "答案：",
                    "Answer:"
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
                retriever=dict(type=retriever, **fixidlist),
                inferencer=get_cot_inferencer(sc_size=SAMPLE_SIZE, fixidlist=fixidlist, cot_prompts=cot_conclude_hint),
            ),
            eval_cfg=dict(evaluator=dict(type=OpsEvalGenMCEvaluator)))
        for shot_abbr, fixidlist, shot_hint_id, retriever in zip(
            ['Zero-shot', '3-shot'],
            [dict(), dict(fix_id_list=[0,1,2])],
            [0, 1],
            [ZeroRetriever, FixKRetriever]
        )
        for qtype, qtype_hint_id in zip(
            ['single', 'multiple'],
            [0, 1]
        )
        for lang, prompt_hint, cot_think_hint, cot_conclude_hint in zip(
            ['en', 'zh'],
            [
                "Here is a multiple-choice question.\n",
                "以下是一个选择题。\n"
            ],
            [
                f"Let's think step by step.\n",
                f"让我们逐个选项分析：\n"
            ],
            [
                [f'{prompts[shot_hint_id][qtype_hint_id][0]}Therefore the answer is: \n'],
                [f'{prompts[shot_hint_id][qtype_hint_id][1]}因此答案是：\n']
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