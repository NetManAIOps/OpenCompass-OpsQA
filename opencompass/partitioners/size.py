import copy
import math
import os.path as osp
from fnmatch import fnmatch
from typing import List, Tuple, Union

import mmengine
from mmengine.config import Config, ConfigDict

from opencompass.registry import PARTITIONERS
from opencompass.utils import (build_dataset_from_cfg, dataset_abbr_from_cfg,
                               get_infer_output_path)

from .base import BasePartitioner


@PARTITIONERS.register_module()
class SizePartitioner(BasePartitioner):
    """Task partitioner based on the size of the dataset (with some rough
    expansion as an estimation of computational cost).

    Args:
        out_dir (str): The output directory of tasks.
        max_task_size (int): The maximum size of a task.
        gen_task_coef (int): The dataset cost measurement coefficient for
            generation tasks.
        dataset_size_path (str): The path to the dataset size cache file.
    """

    def __init__(self,
                 out_dir: str,
                 max_task_size: int = 2000,
                 gen_task_coef: int = 20,
                 dataset_size_path: str = '.cache/dataset_size.json'):
        super().__init__(out_dir)
        self.max_task_size = max_task_size
        self.gen_task_coef = gen_task_coef
        self.dataset_size_path = dataset_size_path

    def partition(self, models: List[ConfigDict], datasets: List[ConfigDict],
                  work_dir: str, out_dir: str) -> List[ConfigDict]:
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
            List[ConfigDict]: A list of tasks.
        """

        datasets = sorted(datasets,
                          key=lambda x: self.get_cost(x),
                          reverse=True)
        tasks = []
        for model in models:
            task = Config({
                'models': [model],
                'datasets': [[]],
                'work_dir': work_dir
            })
            num_data = 0
            for dataset in datasets:
                filename = get_infer_output_path(model, dataset, out_dir)
                root, ext = osp.splitext(filename)
                # skip the task if the task output exists
                if osp.exists(filename):
                    continue
                dataset_size = self.get_cost(dataset)
                if dataset_size > self.max_task_size:
                    dataset_splits = self.split_dataset(dataset)
                    for i, dataset_split in enumerate(dataset_splits):
                        # skip the task it the task output exists
                        if not osp.exists(f'{root}_{i}{ext}'):
                            tasks.append(
                                Config({
                                    'models': [model],
                                    'datasets': [[dataset_split]],
                                    'work_dir': work_dir
                                }))
                else:
                    if num_data + dataset_size > self.max_task_size:
                        tasks.append(task)
                        task = Config({
                            'models': [model],
                            'datasets': [[]],
                            'work_dir': work_dir
                        })
                        num_data = 0
                    task['datasets'][0].append(dataset)
                    num_data = num_data + dataset_size
            if task['datasets'][0]:
                tasks.append(task)

        return tasks

    @property
    def dataset_size(self):
        if not hasattr(self, '_dataset_size'):
            if osp.exists(self.dataset_size_path):
                self._dataset_size = mmengine.load(self.dataset_size_path)
            else:
                self._dataset_size = {}
        return self._dataset_size

    def split_dataset(self, dataset_cfg: ConfigDict) -> List[ConfigDict]:
        """Split dataset into several parts."""
        dataset_size, num_repeats = self.get_cost(dataset_cfg,
                                                  get_raw_factors=True)
        split_configs = []
        abbr = dataset_abbr_from_cfg(dataset_cfg)
        step = self.max_task_size // num_repeats
        # evenly distribute the task
        step = math.ceil(dataset_size / math.ceil(dataset_size / step))
        for part, i in enumerate(range(0, dataset_size, step)):
            cfg = copy.deepcopy(dataset_cfg)
            cfg['abbr'] = abbr + f'_{part}'
            test_range = cfg['reader_cfg'].get('test_range', '')
            cfg['reader_cfg']['test_range'] = f'{test_range}[{i}:{i+step}]'
            split_configs.append(cfg)
        return split_configs

    def get_factor(self, dataset: ConfigDict) -> int:
        infer_cfg = dataset.infer_cfg
        template = (infer_cfg.prompt_template.template if 'prompt_template'
                    in infer_cfg else infer_cfg.ice_template.template)
        # If it's the Gen template, the dataset size will be multiplied by the
        # self.gen_task_coef
        factor = self.gen_task_coef
        # If it's the PPL template, the dataset size will be multiplied by the
        # number of labels
        if isinstance(template, dict):
            ctr = sum(key in template for key in ('begin', 'round', 'end'))
            if ctr != len(template.keys()):
                factor = len(template.keys())

        dataset_abbr = dataset_abbr_from_cfg(dataset)
        if any(
                fnmatch(dataset_abbr, pattern)
                for pattern in ('bbh*', 'gsm8k*', 'math*', 'strategyqa*',
                                'agieval-jec*', 'agieval-gaokao-mathcloze',
                                'agieval-math')):
            factor *= 10

        return factor

    def get_cost(self,
                 dataset: ConfigDict,
                 get_raw_factors: bool = False) -> Union[int, Tuple[int, int]]:
        """Get the computational cost of inferring on the dataset.

        Args:
            dataset (ConfigDict): The dataset config.
            get_raw_factors (bool): If True, the raw factors of computational
                cost will be returned.

        Returns:
            int or Tuple[int, int]: The size of the dataset. If get_raw_factors
                is True, the number of repeats will also be returned.
        """
        dataset_abbr = dataset_abbr_from_cfg(dataset)

        test_range = dataset.reader_cfg.get('test_range', '')
        factor = self.get_factor(dataset)

        if dataset_abbr in self.dataset_size:
            actual_size = eval('len(range(self.dataset_size[dataset_abbr])'
                               f'{test_range})')
            if get_raw_factors:
                return actual_size, factor
            return factor * actual_size

        dataset = build_dataset_from_cfg(dataset)
        self.dataset_size[dataset_abbr] = len(dataset.test)

        mmengine.mkdir_or_exist('.cache/')
        mmengine.dump(self.dataset_size,
                      self.dataset_size_path,
                      indent=4,
                      ensure_ascii=False)

        actual_size = eval('len(range(self.dataset_size[dataset_abbr])'
                           f'{test_range})')
        if get_raw_factors:
            return actual_size, factor
        return factor * actual_size
