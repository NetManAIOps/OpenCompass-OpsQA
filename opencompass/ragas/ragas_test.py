import pandas as pd
import os, json, re, random, time
import importlib
import judge
import config
from judge import *

test_df = pd.DataFrame({
    "id": [0, 1],
    "question": ["长城是哪个朝代的中国皇帝开始建造的？", 
                "Who is considered the father of modern physics?"],
    "ground_truth": ["长城的建造始于春秋战国时期，但最著名的部分是秦始皇命令建造的。",
                "Albert Einstein is often considered the father of modern physics for his development of the theory of relativity."],
    "answer": ["长城的建造始于春秋战国时期，但其中最著名的部分是由秦始皇在公元前221年到公元前206年间开始建造的。",
                "Albert Einstein, known for his theory of relativity and contributions to the development of quantum mechanics, is often considered the father of modern physics."],
})

scores = compute_scores(test_df)
print(scores)