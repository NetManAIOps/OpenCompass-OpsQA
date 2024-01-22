from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever, ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer, SCInferencer, CoTInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.utils.text_postprocessors import first_capital_postprocess_multi
from opencompass.datasets import OwlDataset
with read_base():
    from ...paths import ROOT_DIR

SAMPLE_SIZE = 3

owl_reader_cfg = dict(
    input_columns=['question', 'A', 'B', 'C', 'D', 'category', 'id', 'solution'],
    output_column='answer',
    train_split='dev'
)

owl_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    sc_size=SAMPLE_SIZE, 
    pred_postprocessor=dict(type='owl-choice'))

prompts = [
    # Zero-shot
    [
        '', '请直接给出正确答案的选项。'
    ],
    # 3-shot
    [
        '', ''
    ]
]

owl_path = f'{ROOT_DIR}data/opseval/owl/'

owl_cot = [
    dict(
        type=OwlDataset,
        abbr=f'owl-{shot_abbr}-{lang}-sc+cot',
        path=owl_path, 
        name=f'{lang}',
        reader_cfg=owl_reader_cfg,
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
        eval_cfg=owl_eval_cfg)
        for shot_abbr, fixidlist, shot_hint_id, retriever in zip(
            ['Zero-shot', '3-shot'],
            [dict(fix_id_list=None), dict(fix_id_list=[0,1,2])],
            [0, 1],
            [ZeroRetriever, FixKRetriever]
        )
        for lang, prompt_hint, cot_hint in zip(
            ['en', 'zh'],
            [
                f"Here is a multiple-answer question about {{category}} Operation and Maintainance.\n{{question}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nLet's think step by step.\n",
                f"以下关于{{category}}的选择题。\n{{question}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\n让我们逐个选项分析：\n"
            ],
            [
                [f'{prompts[shot_hint_id][0]}\nTherefore the answer is: \n'],
                [f'{prompts[shot_hint_id][1]}\n因此答案是：\n']
            ]
        )
]


owl_naive = [
    dict(
        type=OwlDataset,
        abbr=f'owl-{shot_abbr}-{lang}-sc',
        path=owl_path, 
        name=f'{lang}',
        reader_cfg=owl_reader_cfg,
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
        eval_cfg=owl_eval_cfg)
        for shot_abbr, fixidlist, shot_hint_id, retriever in zip(
            ['Zero-shot', '3-shot'],
            [dict(fix_id_list=None), dict(fix_id_list=[0,1,2])],
            [0, 1],
            [ZeroRetriever, FixKRetriever]
        )
        for lang, prompt_hint in zip(
            ['en', 'zh'],
            [
                f"Here is a multiple-answer question about {{category}} Operation and Maintainance.{prompts[shot_hint_id][0]}\n{{question}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nAnswer: \n",
                f"以下关于{{category}}的选择题。{prompts[shot_hint_id][1]}\n{{question}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\n答案：\n"
            ],
        )
]

