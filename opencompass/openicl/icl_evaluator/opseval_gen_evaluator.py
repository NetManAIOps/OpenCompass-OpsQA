import random
from typing import List
import numpy as np
from opencompass.registry import ICL_EVALUATORS
from .icl_base_evaluator import BaseEvaluator
import re
from collections import Counter
import jieba
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def extract_answer(text: str) -> str:
    """
    text: 模型原始输出
    return: 提取的答案
    """
    s = text
    s += ' '
    matched = re.search(r'answer.+?[^\w]([a-zA-Z][^\w])', s.lower())
    if matched:
        s = s[matched.span()[0]:]
    if not re.match(r'[A-Z][^\w]', s):
        matched = re.search(r'[^\w][A-Z][^\w]', s)
        if matched:
            s = s[matched.span()[0] + 1:]

    s = s.strip() + ' '
    pattern = r'([A-Z][\s,]+(and)?[\s,]*)*[A-Z][^\w]'
    # matched = re.match(pattern, s, re.S)
    matched = list(re.finditer(pattern, s, re.S))
    first = matched[0].group() if matched else ''
    last = matched[-1].group() if matched else ''

    def process_prefix(prefix):
        ans = list(
            set([
                c.upper() for c in re.split('[^A-Z]', prefix)
                if len(c) == 1 and c.isalpha()
            ]))
        ans.sort()
        return ','.join(ans)
    
    return [process_prefix(a) for a in [first, last]]

def sc_cot(prediction, answer):
    if not isinstance(prediction[0], list):
        prediction = [prediction]
    sc_thoughts = prediction
    assert isinstance(sc_thoughts, list)

    right_count = 0

    sc_choices = []

    # Iterate through sc cases
    for sc_case in sc_thoughts:
        assert isinstance(sc_case, list)
        # Iterate through stage of thoughts:
        thought_choices = []
        for thought in sc_case:
            assert isinstance(thought, str)
            prediction = extract_answer(thought)
            thought_choices += prediction
            if any([d == answer for d in prediction]):    # 所有thought中只要有一个正确即可
                right_count += 1
                thought_choices = [answer]
                break
        # print(Counter(thought_choices).most_common(1)[0][0])
        sc_choices.append(Counter(thought_choices).most_common(1)[0][0])
    try:
        # print(Counter(sc_choices).most_common(1)[0], answer)
        if Counter(sc_choices).most_common(1)[0][0] == answer:
            return 1
    except:
        print(sc_choices)
        print(q)
        raise
    return 0

def not_sc(prediction, answer):
    if not isinstance(prediction[0], list):
        prediction = [prediction]
    sc_thoughts = prediction
    assert isinstance(sc_thoughts, list)

    right_count = 0

    # Iterate through sc cases
    sc_case = random.choice(sc_thoughts)
    # for sc_case in sc_thoughts[:1]:
    assert isinstance(sc_case, list)
    # Iterate through stage of thoughts:
    for thought in sc_case:
        assert isinstance(thought, str)
        prediction = extract_answer(thought)
        if any([d == answer for d in prediction]):    # 所有thought中只要有一个正确即可
            right_count += 1
            break
    if right_count > 0:
        return 1
    return 0

class OpsEvalGenMCEvaluator(BaseEvaluator):
    """
    用于评测OpsEval中的各种选择题
    """
    
    def __init__(self):
        super().__init__()
        
    def score(self, predictions: List, references: List) -> dict:
        tot = len(predictions)
        correct = 0
        sc_correct = 0
        for pred, ans in zip(predictions, references):
            correct += not_sc(pred, ans)
            sc_correct += sc_cot(pred, ans)
        return {
            'Accuracy': correct / tot * 100,
            'SC-Accuracy': sc_correct / tot * 100,
        }

class OpsEvalGenQAEvaluator(BaseEvaluator):
    
    def __init__(self, language='en'):
        super().__init__()
        self.language = language
    
    def score(self, predictions: List, references: List) -> dict:
        tot_bleu, tot_rouge = 0, 0
        for pred, ans in zip(predictions, references):
            bleu_score, rouge_score = self.get_rouge_bleu(pred, ans, self.language)
            tot_bleu += bleu_score
            tot_rouge += rouge_score
        return {
            "bleu": tot_bleu / len(predictions),
            "rouge": tot_rouge / len(predictions)
        }
    
    def get_rouge_bleu(self, pred, answer, language='en'):
        rouge = Rouge()
        smoothie = SmoothingFunction().method7
        def clean_word(words):
            punctuations = """！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～、。，·ˉˇ¨〃‘’“”々〆〇〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."""
            return [word for word in words if word.strip() and word not in punctuations]
        try:
            if language == 'en':
                bleu_score = sentence_bleu([clean_word(answer.split())], clean_word(pred.split()), smoothing_function=smoothie)
                rouge_score = rouge.get_scores(' '.join(clean_word(pred.split())), ' '.join(clean_word(answer.split())), avg=True)['rouge-l']['f']
            else:
                answer_tokenized = clean_word(list(jieba.cut(answer)))
                pred_tokenized = clean_word(list(jieba.cut(pred)))
                bleu_score = sentence_bleu([answer_tokenized], pred_tokenized, smoothing_function=smoothie)
                rouge_score = rouge.get_scores(' '.join(pred_tokenized), ' '.join(answer_tokenized), avg=True)['rouge-l']['f']
        except Exception as err:
            print(f"[WARNING] Error when calculating bleu and rouge: {err}")
            bleu_score, rouge_score = 0.0, 0.0
        return bleu_score, rouge_score

class OpsEvalRagasEvaluator(BaseEvaluator):

    def __init__(self):
        super().__init__()

    def score(self, predictions: List, references: List) -> dict:
        from opencompass.ragas.judge import calculate_scores
        reference = {"id": idx, "answer": ref
                            for idx, ref in enumerate(references)}
        answers = {"id": idx, "answer": ans
                            for idx, ans in enumerate(predictions)}
        report = calculate_scores(reference, answers)
        return report