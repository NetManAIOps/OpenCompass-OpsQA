from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever, ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer, SCInferencer, CoTInferencer
from opencompass.utils.text_postprocessors import first_capital_postprocess_multi
from opencompass.datasets import OracleDataset
from mmengine.config import read_base
with read_base():
    from ...commons.cfgs import choice_qa_reader_cfg, choice_single_eval_cfg
    from ...commons.prompts import prompts
    from ...paths import ROOT_DIR

companies = [
    'dfcdata',
    'huaweicloud',
    'pufa',
    'rzy',
    'lenovo',
    'bosc',
    'gtja',
    'zabbix'
]

SAMPLE_SIZE = 3

company_cot = [
    dict(
        type=OracleDataset,
        abbr=f'{company}-{shot_abbr}-{lang}-sc+cot',
        path=f'{ROOT_DIR}data/opseval/{company}/', 
        name=f'{company}_{lang}',
        reader_cfg=choice_qa_reader_cfg,
        infer_cfg=dict(
            ice_template=dict(
                type=PromptTemplate,
                template=dict( 
                    round=[
                        dict(
                            role="HUMAN",
                            prompt=prompt_hint
                        ),
                        dict(role="BOT", prompt=f"{{solution}}\n"),
                        dict(
                            role="HUMAN",
                            prompt=cot_hint[0]
                        ),
                        dict(
                            role="BOT", prompt=f"{{answer}}\n",
                        )
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
            retriever=dict(type=retriever),
            inferencer=dict(
                type=CoTInferencer,
                save_every=20,
                cot_prompts=cot_hint,
                generation_kwargs=dict(temperature=0.7),
                infer_type='SC',
                sc_size = SAMPLE_SIZE, 
                max_out_len = 400,
                **fixidlist
            ),
        ),
        eval_cfg=choice_single_eval_cfg)
        for shot_abbr, fixidlist, shot_hint_id, retriever in zip(
            ['Zero-shot', '3-shot'],
            [dict(fix_id_list=None), dict(fix_id_list=[0,1,2])],
            [0, 1],
            [ZeroRetriever, FixKRetriever]
        )
        for lang, prompt_hint, cot_hint in zip(
            # ['en', 'zh'],
            # [
            #     f"Here is a multiple-answer question about Oracle database.\n{{question}}\nLet's think step by step.\n",
            #     f"以下关于Oracle数据库的选择题。\n{{question}}\n让我们逐个选项分析：\n"
            # ],
            # [
            #     [f'{prompts[shot_hint_id][qtype_hint_id][0]}\nTherefore the answer is: \n'],
            #     [f'{prompts[shot_hint_id][qtype_hint_id][1]}\n因此答案是：\n']
            # ]
            ['zh'],
            [
                # f"Here is a multiple-answer question about Oracle database.\n{{question}}\nLet's think step by step.\n",
                f"以下关于Oracle数据库的选择题。\n{{question}}\n让我们逐个选项分析：\n"
            ],
            [
                # [f'{prompts[shot_hint_id][qtype_hint_id][0]}\nTherefore the answer is: \n'],
                [f'{prompts[shot_hint_id][0][1]}\n因此答案是：\n']
            ]
        )
    for company in companies
]

company_naive = [
    dict(
        type=OracleDataset,
        abbr=f'{company}-{shot_abbr}-{lang}-sc',
        path=f'{ROOT_DIR}data/opseval/{company}/', 
        name=f'{company}_{lang}',
        reader_cfg=choice_qa_reader_cfg,
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
            retriever=dict(type=retriever),
            inferencer=dict(
                type=SCInferencer,
                save_every=20,
                generation_kwargs=dict(temperature=0.7),
                infer_type='SC',
                sc_size = SAMPLE_SIZE,  
                max_out_len = 400,
                **fixidlist
            ),
        ),
        eval_cfg=choice_single_eval_cfg)
        for shot_abbr, fixidlist, shot_hint_id, retriever in zip(
            ['Zero-shot', '3-shot'],
            [dict(fix_id_list=None), dict(fix_id_list=[0,1,2])],
            [0, 1],
            [ZeroRetriever, FixKRetriever]
        )
        for lang, prompt_hint in zip(
            # ['en', 'zh'],
            ['zh'],
            # [
            #     f"Here is a multiple-answer question.\n{prompts[shot_hint_id][0][0]}\n{{question}}\nAnswer: \n",
            #     f"回答下面的选择题。\n{prompts[shot_hint_id][0][1]}\n{{question}}\n答案：\n"
            # ],
            [
                # f"Here is a multiple-answer question.\n{prompts[shot_hint_id][0][0]}\n{{question}}\nAnswer: \n",
                f"回答下面的选择题。\n{prompts[shot_hint_id][0][1]}\n{{question}}\n答案：\n"
            ],
        )
    for company in companies
]
