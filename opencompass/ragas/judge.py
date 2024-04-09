import os
import json
import logging
import argparse
import pandas as pd
from config import config, load_llm, load_embeddings
from datasets import Dataset

logger = logging.getLogger(__name__)

judge_config = config.get('judge', {})


def validate_and_format_answers(answers: list[dict]) -> list[dict]:
    """
    Validate the structure of the team's answers and reformat them for further processing.

    Parameters:
        answers: A list of dictionaries, each containing an 'id' and an 'answer'.

    Returns:
        A reformatted list of dictionaries with validated 'id' and 'answer' keys.

    Raises:
        AssertionError: If any of the dictionaries in the input list does not have the expected structure.
    """
    validated_answers = []
    for item in answers:
        assert isinstance(item, dict), "Each item must be a dictionary."
        assert 'id' in item and 'answer' in item, "Each dictionary must have 'id' and 'answer' keys."
        assert isinstance(item['id'], int), "The 'id' must be an integer."
        
        extra = {}
        if item.get('label') is not None:
            label = int(item['label'])
            assert label in [0, 1]
            extra['label'] = label

        validated_answers.append({
            'id': item['id'],
            'answer': item['answer'],
            **extra,
        })
    return validated_answers


def preprocess_data(ground_truth: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the ground truth and prediction data for scoring.

    Parameters:
        ground_truth: A DataFrame containing the ground truth answers with columns 'id' and 'answer'.
        predictions: A DataFrame containing the predicted answers with columns 'id' and 'answer'.

    Returns:
        A merged DataFrame with columns 'id', 'ground_truth', and 'answer' for scoring.
    """
    ground_truth['answer'] = ground_truth['answer'].astype('str')
    ground_truth = ground_truth.rename(columns={
        'query': 'question',
        'answer': 'ground_truth',
    })
    return ground_truth.merge(predictions, on='id', how='left').fillna('')


def calculate_score(reference: list[dict], answers: list[dict]) -> dict:
    """
    Calculate the score of the team's answers based on the reference answers.

    Parameters:
        reference: A list of dictionaries, each containing an 'id', a 'query', and the ground_truth 'answer'.
            Example: [{"id": 1, "query": "What is AI?", "answer": "Artificial Intelligence"}]
        answers: A list of dictionaries, each containing an 'id', a 'query', and the team's 'answer'.
            Example: [{"id": 1, "query": "What is AI?", "answer": "Study of intelligent agents"}]

    Returns:
        A dictionary containing the total score and the detail of scores per question.
    """
    gt_df = pd.DataFrame(reference)
    preds_df = pd.DataFrame(validate_and_format_answers(answers))
    data = preprocess_data(gt_df, preds_df)
    res_df = compute_scores(data)
    detail = res_df.to_dict(orient='records')

    overall_score = sum([item['score'] for item in detail]) / len(detail)
    accuracy = sum([item['correct'] for item in detail]) / len(detail)

    report = {
        'score': overall_score,
        'accuracy': accuracy,
        "detail": detail,
    }

    if 'label' in res_df:
        # Evaluate consistency with human evaluation
        import numpy as np
        from scipy.stats import pearsonr
        from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score

        gt_score = res_df['label']
        pred_score = res_df['score']
        correlation_coefficient, _ = pearsonr(pred_score, gt_score)
        auc = roc_auc_score(gt_score, pred_score)
        thresholds = np.linspace(0, 1, 101)
        best_f1 = 0
        best_threshold = 0
        for t in thresholds:
            pred_binary = (pred_score >= t).astype(int)
            f1 = f1_score(gt_score, pred_binary)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t

        pred_binary_best = (pred_score >= best_threshold).astype(int)
        accuracy = accuracy_score(gt_score, pred_binary_best)
        precision = precision_score(gt_score, pred_binary_best)
        recall = recall_score(gt_score, pred_binary_best)
        misclassified_questions = list(res_df[pred_binary_best != gt_score]['id'])

        report['consistency'] = {
            'correlation_coefficient': correlation_coefficient,
            'auc': auc,
            'best_f1': best_f1,
            'best_threshold': best_threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'misclassified_questions': misclassified_questions,
        }

    return report

def compute_scores(df: pd.DataFrame) -> list[dict]:
    langsmith_config = config.get('langsmith', {})
    langsmith_enabled = langsmith_config.get('enabled', False)

    if langsmith_enabled:
        os.environ["LANGCHAIN_TRACING_V2"] = 'true'
        os.environ["LANGCHAIN_API_KEY"] = langsmith_config.get('api_key')
        os.environ["LANGCHAIN_ENDPOINT"] = langsmith_config.get('endpoint')

    from ragas import evaluate, RunConfig
    from metric import answer_correctness
    from langchain.callbacks.tracers import LangChainTracer
    
    new_df = df.copy()
    for col in ['question', 'ground_truth', 'answer']:
        new_df[col] = new_df[col].apply(lambda s: json.dumps(s, ensure_ascii=False))
    dataset = Dataset.from_pandas(new_df)
    callbacks = []

    if langsmith_enabled:
        tracer = LangChainTracer(project_name=langsmith_config.get('project_name'))
        callbacks.append(tracer)

    result = evaluate(
        dataset,
        metrics=[
            answer_correctness,
        ],
        llm=load_llm(),
        embeddings=load_embeddings(),
        run_config=RunConfig(max_workers=judge_config.get('max_workers', 16)),
        callbacks=callbacks,
    )

    res_df = result.to_pandas()
    res_df = res_df[['id', 'answer_correctness']].rename(columns={
        'answer_correctness': 'score',
    })
    
    res_df['score'] = res_df['score'].fillna(0)

    correct_threshold = judge_config.get('correct_threshold', 0.7)
    res_df['correct'] = res_df['score'] >= correct_threshold

    return res_df.merge(df, on='id', how='left')


def read_jsonl(file_path: str) -> list[dict]:
    """
    Read a .jsonl file and return its contents as a list of dictionaries.

    Parameters:
        file_path: The path to the .jsonl file to be read.

    Returns:
        A list of dictionaries, each representing a line in the .jsonl file.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate the score of answers based on reference answers.')
    parser.add_argument('-d', '--directory', type=str, default='',
                        help='Working directory to prefix file paths. Default: "" (current directory)')
    parser.add_argument('-r', '--reference_file', type=str, default='ground_truth.jsonl',
                        help='Path to the reference file containing ground truth answers. Default: test/ground_truth.jsonl')
    parser.add_argument('-a', '--answer_file', type=str, default='result.jsonl',
                        help='Path to the answer file containing team\'s answers. Default: test/result.jsonl')
    parser.add_argument('-o', '--output', type=str, default='report.json',
                        help='Path to the output file where the results will be saved in JSON format. If not provided, results will only be printed to stdout.')

    args = parser.parse_args()

    reference_file_path = os.path.join(args.directory, args.reference_file)
    answer_file_path = os.path.join(args.directory, args.answer_file)
    output_file_path = os.path.join(args.directory, args.output)

    # Read the reference and answer files
    reference = read_jsonl(reference_file_path)
    answers = read_jsonl(answer_file_path)

    # Calculate the score
    report = calculate_score(reference, answers)

    # Output the overall stats
    print(f"Score    : {round(report['score'], 4)}")
    print(f"Accuracy : {round(report['accuracy'], 4)}")

    if report.get('consistency'):
        print('\nConsistency with human evaluation:')
        c = report['consistency']
        print(f"  Correlation coefficient: {round(c['correlation_coefficient'], 4)}")
        print(f"  AUC Score    : {round(c['auc'], 4)}")
        print(f"  Best F1 Score: {round(c['best_f1'], 4)} (T={round(c['best_threshold'], 4)})")
        print(f"  Assuming T={round(c['best_threshold'], 4)}:")
        print(f"    - Accuracy : {round(c['accuracy'], 4)}")
        print(f"    - Precision: {round(c['precision'], 4)}")
        print(f"    - Recall   : {round(c['recall'], 4)}")

    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
