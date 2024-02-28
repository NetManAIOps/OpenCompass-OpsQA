from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever, ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer, SCInferencer, CoTInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.utils.text_postprocessors import first_capital_postprocess_multi
from opencompass.datasets import QADataset

tencent_reader_cfg = dict(
    input_columns=['question'],
    output_column='answer',
    train_split='dev'
)

tencent_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator))

tencent_datasets = [
    dict(
        type=QADataset,
        abbr=f'tencent_qa-{shot_abbr}',
        path=f'/mnt/mfs/opsgpt/evaluation/opseval/tencent/', 
        name=f'tencent_zh',
        reader_cfg=tencent_reader_cfg,
        infer_cfg=dict(
            ice_template=dict(
                type=PromptTemplate,
                template=dict( 
                    round=[
                        dict(
                            role="HUMAN",
                            prompt=f"你是一名运维专家，请回答下面这个问题：\n{{question}}\n"
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
                            prompt=f"你是一名运维专家，请回答下面这个问题：\n{{question}}\n"
                        ),
                        dict(role="BOT", prompt="{answer}\n")
                    ]
                ),
                ice_token="</E>",
            ),
            retriever=dict(type=retriever),
            inferencer=dict(
                type=GenInferencer,
                save_every=20,
                generation_kwargs=dict(temperature=0),
                max_out_len = 400,
                **fixidlist
            ),
        ),
        eval_cfg=tencent_eval_cfg)
        for shot_abbr, fixidlist, shot_hint_id, retriever in zip(
            ['Zero-shot', '3-shot'],
            [dict(fix_id_list=None), dict(fix_id_list=[0,1,2])],
            [0, 1],
            [ZeroRetriever, FixKRetriever]
        )
]

