import os.path as osp
from typing import Dict, List

from mmengine.config import Config, ConfigDict

from opencompass.registry import PARTITIONERS
from opencompass.utils import get_infer_output_path

from .base import BasePartitioner


@PARTITIONERS.register_module()
class NaivePartitioner(BasePartitioner):
    """Naive task partitioner. This partitioner will generate a task for each
    model-dataset pair.

    Args:
        config (ConfigDict): The full config dict.
    """

    def partition(self, models: List[ConfigDict], datasets: List[ConfigDict],
                  work_dir: str, out_dir: str) -> List[Dict]:
        """Partition model-dataset pairs into tasks. Each task is defined as a
        dict and will run independently as a unit. Its structure is as
        follows:

        .. code-block:: python

            {
                'models': [],  # a list of model configs
                'datasets': [[]],  # a nested list of dataset configs, each
                                    list corresponds to a model
                'work_dir': '',  # the work dir
            }

        Args:
            models (List[ConfigDict]): A list of model configs.
            datasets (List[ConfigDict]): A list of dataset configs.
            work_dir (str): The work dir for the task.
            out_dir (str): The full output path for the task, intended for
                Partitioners to check whether the task is finished via the
                existency of result file in this directory.

        Returns:
            List[Dict]: A list of tasks.
        """

        tasks = []
        for dataset in datasets:
            for model in models:
                filename = get_infer_output_path(model, dataset, out_dir)
                if osp.exists(filename):
                    continue
                task = Config({
                    'models': [model],
                    'datasets': [[dataset]],
                    'work_dir': work_dir
                })
                tasks.append(task)
        return tasks
