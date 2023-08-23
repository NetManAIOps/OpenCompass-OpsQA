from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever, ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer, SCInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.utils.text_postprocessors import first_capital_postprocess_multi
from opencompass.datasets import OReillyChoiceDataset, OReillyEvaluator, oreilly_choice_postprocess, OReillyDataset

SAMPLE_SIZE = 3

oreilly_reader_cfg = dict(
    input_columns=['topic','question','choices','qtype'],
    output_column='answer',
    train_split='dev'
)

oreilly_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    sc_size=SAMPLE_SIZE,
    pred_postprocessor=dict(type='oreilly-choice'))
    

oreilly_datasets = [
    dict(
        type=OReillyDataset,
        abbr=f'zh-en-comparison-zeroshot-sc-en',
        path='/mnt/mfs/opsgpt/evaluation/translated', 
        name=f'original',
        qtype=0,
        # sample_setting=dict(
        #     sample_size=1
        # ), 
        reader_cfg=oreilly_reader_cfg,
        infer_cfg=dict(
            ice_template=dict(
                type=PromptTemplate,
                template=dict(
                    begin="</E>", 
                    round=[
                        dict(
                            role="HUMAN",
                            prompt=f"Here is a {{qtype}} question about {{topic}}.\n{{question}}\n{{choices}}\nAnswer:\n"
                        ),
                        dict(role="BOT", prompt="{answer}\n")
                    ]
                ),
                ice_token="</E>",
            ),
            retriever=dict(type=FixKRetriever),
            inferencer=dict(
                type=SCInferencer,
                save_every=100, 
                generation_kwargs=dict(temperature=0.7),
                infer_type='SC',
                sc_size = SAMPLE_SIZE, 
                fix_id_list=[0,1,2]
            ),
        ),
        eval_cfg=oreilly_eval_cfg
    ), 
    dict(
        type=OReillyDataset,
        abbr=f'zh-en-comparison-zeroshot-sc-zh',
        path='/mnt/mfs/opsgpt/evaluation/translated', 
        name=f'chinese',
        qtype=0,
        # sample_setting=dict(
        #     sample_size=1
        # ), 
        reader_cfg=oreilly_reader_cfg,
        infer_cfg=dict(
            ice_template=dict(
                type=PromptTemplate,
                template=dict(
                    begin="</E>", 
                    round=[
                        dict(
                            role="HUMAN",
                            prompt=f"以下是关于{{topic}}的单项选择题，请选出其中的正确答案。\n{{question}}\n{{choices}}\n答案：\n"
                        ),
                        dict(role="BOT", prompt="{answer}\n")
                    ]
                ),
                ice_token="</E>",
            ),
            retriever=dict(type=FixKRetriever),
            inferencer=dict(
                type=SCInferencer,
                save_every=100, 
                generation_kwargs=dict(temperature=0.7),
                infer_type='SC',
                sc_size = SAMPLE_SIZE, 
                fix_id_list=[0,1,2]
            ),
        ),
        eval_cfg=oreilly_eval_cfg
    ), 
]

