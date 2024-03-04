import os

model_name = os.environ.get('MODEL')
model_path = os.environ.get('MODEL_PATH')
datasets = os.environ.get('DATASETS')
if model_name is None:
    raise NotImplementedError("Model name is none!")
if model_path is None:
    raise NotImplementedError("Model path is none!")
if datasets is None:
    raise NotImplementedError("Datasets is none!")
model_abbr = f"nm_qwen_14b_{model_name.replace('-', '_')}"

with open('/mnt/tenant-home_speed/lyh/opseval-configs/xz/runconfig_base.py') as fi, open('/mnt/tenant-home_speed/lyh/opseval-configs/xz/runconfig.py', 'wt') as fo:
    content = fi.read().replace('$MODEL$', model_name).replace('$MODEL_ABBR$', model_abbr).replace('$MODEL_PATH$', model_path).replace('$DATASETS$', datasets)
    fo.write(content)
