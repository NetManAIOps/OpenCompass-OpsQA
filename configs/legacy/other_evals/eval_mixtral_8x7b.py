from mmengine.config import read_base

with read_base():
    from .datasets.collections.base_medium_llama import piqa_datasets, siqa_datasets
<<<<<<< HEAD:configs/legacy/other_evals/eval_mixtral_8x7b.py
    from .models.mixtral.mixtral_8x7b_32k import models
=======
    from .models.llama.llama2_7b import models
>>>>>>> upstream/main:configs/eval_llama2_7b.py


datasets = [*piqa_datasets, *siqa_datasets]