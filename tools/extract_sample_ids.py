import csv  # noqa
import json  # noqa
import os  # noqa
import re  # noqa
import sys  # noqa
from os import path as osp  # noqa

from tqdm import tqdm  # noqa

from opencompass.datasets import oreilly_choice_postprocess as extract_answer

default_folder = '/mnt/mfs/opsgpt/opencompass/outputs/archive'


def get_sampled_ids(folder, model):
    sample_ids = []
    prediction_folder = osp.join(default_folder, folder, 'predictions', model)
    for json_file in map(lambda x: osp.join(prediction_folder, x),
                         os.listdir(prediction_folder)):
        if not osp.isfile(json_file):
            continue
        with open(json_file) as f:
            data = json.load(f)
        for idx, row in data.items():
            sample_ids.append(row['reference']['id'])
    return set(sample_ids)


def extract_samples(folder, model, sample_ids):
    """
    folder: outputs/archive文件夹下的子文件夹名
    model: predictions文件夹中的model选择
    sample_ids: 从其他评测结果中提取的采样id集合
    这个函数能从对应结果目录中提取出约5000道题的子集
    """
    prediction_folder = osp.join(default_folder, folder, 'predictions', model)
    scores = []
    sample_scores = []
    new_folder = osp.join(default_folder, folder + '-sample', 'predictions',
                          model)
    os.makedirs(new_folder, exist_ok=True)
    for filename in tqdm(os.listdir(prediction_folder)):
        json_file = osp.join(prediction_folder, filename)
        if not osp.isfile(json_file):
            continue
        with open(json_file) as f:
            data = json.load(f)
        tot = len(data)
        correct_cnt = 0
        sample_correct_cnt = 0
        tot_sample = 0
        samples = dict()
        for idx, row in data.items():
            pid = row['reference']['id']
            gold = row['reference']['answer']
            pred = extract_answer(row['prediction'])
            if gold == pred:
                correct_cnt += 1
            if pid in sample_ids:
                samples[idx] = row
                tot_sample += 1
                if gold == pred:
                    sample_correct_cnt += 1
        with open(osp.join(new_folder, filename), 'w') as f:
            json.dump(samples, f, indent=4)

        scores.append(correct_cnt / tot * 100)
        sample_scores.append(
            (filename.replace('.json',
                              ''), sample_correct_cnt * 100 / tot_sample))
    print(sum(scores) / len(scores))
    export_summary_csv(folder, model, sample_ids, sample_scores)


def export_summary_csv(folder, model, sample_ids, scores):
    summary_folder = osp.join(default_folder, folder + '-sample', 'summary')
    os.makedirs(summary_folder, exist_ok=True)
    with open(osp.join(summary_folder, f'summary_{model}.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', model])
        for dataset, score in scores:
            writer.writerow([dataset, score])


if __name__ == '__main__':
    sample_ids = get_sampled_ids('baichuan-oreilly-sample',
                                 'baichuan-13b-chat')
    # sample_ids_2 = extract_sample_ids("glmllama-oreilly-sample",
    #     "chatglm2-6b")
    # assert sample_ids == sample_ids_2
    extract_samples('llama2chat13b-oreilly-full', 'llama-2-13b-chat',
                    sample_ids)
