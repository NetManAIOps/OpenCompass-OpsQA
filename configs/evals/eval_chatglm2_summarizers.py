from mmengine.config import read_base

with read_base():
    # choose a list of datasets
    from ..datasets.collections.base_small import datasets
    # choose a model of interest
    from ..local_models.chatglm2_6b import models
    # and output the results in a choosen format
    from ..summarizers.small import summarizer

    