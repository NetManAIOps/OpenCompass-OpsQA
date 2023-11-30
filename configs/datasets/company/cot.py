from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever, ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer, SCInferencer, CoTInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.utils.text_postprocessors import first_capital_postprocess_multi
from opencompass.datasets import OracleDataset

SAMPLE_SIZE = 5

company_reader_cfg = dict(
    input_columns=['question'],
    output_column='answer',
    train_split='dev'
)

company_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    sc_size=SAMPLE_SIZE, 
    pred_postprocessor=dict(type='oracle-choice'))

prompts = [
    # Zero-shot
    [
        # Single
        ['', '请直接给出正确答案的选项。'],
        # Multiple
        ['You should select all appropriate option letters separated by commas to answer this question. Example of a possible answer: B,C.\n', 
              '请直接给出所有正确答案的选项并用英文逗号分隔，例如：“B,C”。']
    ],
    # 3-shot
    [
        # Single
        ['', ''],
        ['', '']
    ]
]

companies = [
    'dfcdata',
    'huaweicloud',
    'pufa'
]

company_datasets = [
    dict(
        type=OracleDataset,
        abbr=f'{company}-{shot_abbr}-{lang}-sc+cot',
        path=f'/mnt/mfs/opsgpt/evaluation/opseval/{company}/', 
        name=f'{company}_{lang}',
        reader_cfg=company_reader_cfg,
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
        eval_cfg=company_eval_cfg)
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