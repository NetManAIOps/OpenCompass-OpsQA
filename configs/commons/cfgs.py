from opencompass.openicl.icl_evaluator import AccEvaluator

SAMPLE_SIZE = 3

choice_qa_reader_cfg = dict(
    input_columns=['question'],
    output_column='answer',
    train_split='dev'
)

choice_single_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    sc_size=SAMPLE_SIZE, 
    pred_postprocessor=dict(type='oracle-choice'))

mc_abcd_reader_cfg = dict(
    input_columns=['question','A', 'B', 'C', 'D'],
    output_column='answer',
    train_split='dev'
)

qa_gen_reader_cfg = dict(
    input_columns=['question'],
    output_column='answer',
    train_split='dev'
)

qa_ppl_reader_cfg = dict(
    input_columns=['question', 'answer'],
    output_column='answer',
    train_split='dev'
)