from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever, ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer, SCInferencer, CoTInferencer, PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.utils.text_postprocessors import first_capital_postprocess_multi
from opencompass.datasets import OracleDataset
from mmengine import read_base
with read_base():
    from ...commons.cfgs import choice_qa_reader_cfg, choice_single_eval_cfg
    from ...commons.prompts import prompts
    from ...paths import ROOT_DIR

SAMPLE_SIZE = 3

zte_path = f'{ROOT_DIR}data/opseval/zte/'

zte_cot = [
    dict(
        type=OracleDataset,
        abbr=f'zte-{shot_abbr}-{lang}-{qtype}-sc+cot',
        path=zte_path, 
        name=f'zte_{lang}_{qtype}',
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
        for qtype, qtype_hint_id in zip(
            ['single', 'multiple'],
            [0, 1]
        )
        for lang, prompt_hint, cot_hint in zip(
            ['en', 'zh'],
            [
                f"Here is a multiple-answer question about 5G.\n{{question}}\nLet's think step by step.\n",
                f"以下关于5G通信的选择题。\n{{question}}\n让我们逐个选项分析：\n"
            ],
            [
                [f'{prompts[shot_hint_id][qtype_hint_id][0]}\nTherefore the answer is: \n'],
                [f'{prompts[shot_hint_id][qtype_hint_id][1]}\n因此答案是：\n']
            ]
        )
        
]


zte_naive = [
    dict(
        type=OracleDataset,
        abbr=f'zte-{shot_abbr}-{lang}-{qtype}-sc',
        path=zte_path, 
        name=f'zte_{lang}_{qtype}',
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
        for qtype, qtype_hint_id in zip(
            ['single', 'multiple'],
            [0, 1]
        )
        for lang, prompt_hint in zip(
            ['en', 'zh'],
            [
                f"Here is a multiple-answer question about 5G.\n{prompts[shot_hint_id][qtype_hint_id][0]}\n{{question}}\nAnswer: \n",
                f"以下关于5G通信的选择题。\n{prompts[shot_hint_id][qtype_hint_id][1]}\n{{question}}\n答案：\n"
            ],
        )
]

zte_ppl = [
    dict(
        type=OracleDataset,
        abbr=f'zte-{shot_abbr}-{lang}-{qtype}-sc-ppl',
        path=zte_path, 
        name=f'zte_{lang}_{qtype}_ppl',
        # reader_cfg=choice_qa_reader_cfg,
        reader_cfg=dict(
            input_columns=['question','A', 'B', 'C', 'D'],
            output_column='answer',
            train_split='dev'
        ),
        infer_cfg=dict(
            ice_template=dict(
                type=PromptTemplate,
                template={
                    'A' : prompt_hint+f'\n{{question}}\nAnswer: A\n',
                    'B' : prompt_hint+f'\n{{question}}\nAnswer: B\n',
                    'C' : prompt_hint+f'\n{{question}}\nAnswer: C\n',
                    'D' : prompt_hint+f'\n{{question}}\nAnswer: D\n'
                }
                # template={
                #     chr(ord('A')+cid): prompt_hint+' {{question}} '+chr(ord('A')+cid)+': {{choices['+cid+']}}' for cid in enumerate(choices)
                # }
            ),
            prompt_template=dict(
                type=PromptTemplate,
                template={
                    'A' : "</E>" + prompt_hint+'\n{question}\nAnswer: A\n',
                    'B' : "</E>" + prompt_hint+'\n{question}\nAnswer: B\n',
                    'C' : "</E>" + prompt_hint+'\n{question}\nAnswer: C\n',
                    'D' : "</E>" + prompt_hint+'\n{question}\nAnswer: D\n'
                },
                ice_token="</E>",
            ),
            retriever=dict(type=retriever),
            inferencer=dict(
                type=PPLInferencer,
                save_every=20,
                generation_kwargs=dict(temperature=0.7),
                infer_type='SC',
                sc_size = SAMPLE_SIZE,  
                max_out_len = 400,
                **fixidlist
            ),
        ),
        eval_cfg=dict(evaluator=dict(type=AccEvaluator)))
        for shot_abbr, fixidlist, shot_hint_id, retriever in zip(
            ['Zero-shot', '3-shot'],
            [dict(fix_id_list=None), dict(fix_id_list=[0,1,2])],
            [0, 1],
            [ZeroRetriever, FixKRetriever]
        )
        for qtype, qtype_hint_id in zip(
            ['single'],
            [0]
        )
        for lang, prompt_hint in zip(
            ['en', 'zh'],
            [
                f"Here is a multiple-answer question about 5G.\n{prompts[shot_hint_id][qtype_hint_id][0]}",
                f"以下关于5G通信的选择题。\n{prompts[shot_hint_id][qtype_hint_id][1]}"
            ],
        )
]
