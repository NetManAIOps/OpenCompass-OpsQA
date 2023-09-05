from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever, ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer, SCInferencer, CoTInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.utils.text_postprocessors import first_capital_postprocess_multi
from opencompass.datasets import OReillyChoiceDataset, OReillyEvaluator, oreilly_choice_postprocess, OReillyDataset

SAMPLE_SIZE = 5

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
        abbr=f'network-zh-zeroshot-sc+cot-{qtype_abbr}',
        path='/mnt/mfs/opsgpt/evaluation/translated/v3', 
        name=f'ch_{qtype_abbr}',
        qtype=qtype_id,
        # sample_setting=dict(
        #     sample_size=1
        # ), 
        reader_cfg=oreilly_reader_cfg,
        infer_cfg=dict(
            ice_template=dict(
                type=PromptTemplate,
                template=dict(
                    round=[
                        dict(
                            role="HUMAN",
                            prompt=f"下面是一道关于{{topic}}的{qtype_hint}题，\n{{question}}\n{{choices}}\n让我们逐个选项分析：\n"
                        ),
                        dict(role="BOT", prompt="{answer}\n")
                    ]
                ),
                ice_token="</E>",
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(
                type=CoTInferencer,
                save_every=200, 
                cot_prompts=[f'{hint}因此答案是：  \n'], 
                generation_kwargs=dict(temperature=0.7),
                infer_type='SC',
                sc_size = SAMPLE_SIZE,
                max_out_len=max_out_len, 
            ),
        ),
        eval_cfg=oreilly_eval_cfg)
    for qtype_abbr, qtype_id, hint, max_out_len, qtype_hint in zip(
        ['single', 'multiple'],
        [0, 1],
        ['', '请选择所有正确的答案并用英文逗号分隔，例如：“B,C”。\n'], 
        [200, 200],
        ['单选', '多选']
    )
]

