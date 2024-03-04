import random
from typing import List

import evaluate
import numpy as np

from opencompass.registry import ICL_EVALUATORS

from .icl_base_evaluator import BaseEvaluator

class MeanEvaluator(BaseEvaluator):
    
    def __init__(self, field_name) -> None:
        self.field_name = field_name
        super().__init__()
        
    def score(self, predictions: List, references: List) -> dict:
        try:
            metric_result = [pred[self.field_name] for pred in predictions]
        except:
            print(predictions)
            raise
        return {f'mean-{self.field_name}': np.mean(metric_result)}