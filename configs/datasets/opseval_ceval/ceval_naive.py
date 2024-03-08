from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever, ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer, SCInferencer, CoTInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.utils.text_postprocessors import first_capital_postprocess_multi
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.datasets import OracleDataset
with read_base():
    from ...commons.cfgs import choice_qa_reader_cfg, choice_single_eval_cfg
    from ...commons.prompts import prompts
    from ...paths import ROOT_DIR

SAMPLE_SIZE = 3

ceval_path = f'{ROOT_DIR}data/opseval/ceval/'

ceval_naive = [
    dict(
        type=OracleDataset,
        abbr=f'ceval-{shot_abbr}-{lang}-{qtype}-sc',
        path=ceval_path, 
        name=f'ceval_{lang}_{qtype}_ppl',
        # reader_cfg=choice_qa_reader_cfg,
        reader_cfg=dict(
            input_columns=['question','A', 'B', 'C', 'D'],
            output_column='answer',
            train_split='dev'
        ),
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
        eval_cfg=dict(
            evaluator=dict(type=AccEvaluator),
            sc_size=SAMPLE_SIZE, 
            pred_postprocessor=dict(type='oracle-choice')))
        for shot_abbr, fixidlist, shot_hint_id, retriever in zip(
            ['Zero-shot', '3-Shot'],
            [dict(fix_id_list=None), dict(fix_id_list=[0,1,2])],
            [0, 1],
            [ZeroRetriever, FixKRetriever]
        )
        for qtype, qtype_hint_id in zip(
            ['single'],
            [0]
        )
        for lang, prompt_hint in zip(
            ['zh'],
            [
                f"以下是选择题。{prompts[shot_hint_id][qtype_hint_id][1]}\n{{question}}\n答案：\n"
            ],
        )
]