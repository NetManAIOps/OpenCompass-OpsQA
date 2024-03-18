from mmengine.config import read_base

with read_base():
<<<<<<< HEAD:configs/legacy/eval_demo.py
    from ..datasets.others.winograd.winograd_ppl import winograd_datasets
    from ..datasets.others.siqa.siqa_gen import siqa_datasets
=======
    from .datasets.siqa.siqa_gen import siqa_datasets
    from .datasets.winograd.winograd_ppl import winograd_datasets
    from .models.opt.hf_opt_125m import opt125m
    from .models.opt.hf_opt_350m import opt350m
>>>>>>> upstream/main:configs/eval_demo.py

datasets = [*siqa_datasets, *winograd_datasets]
models = [opt125m, opt350m]
