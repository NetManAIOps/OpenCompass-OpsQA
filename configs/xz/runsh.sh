echo `date`, Setup the environment ...
set -e  # exit if error

MODEL="3gpp-sft_lora_sharegpt"

cd /mnt/tenant-home_speed/lyh/OpenCompass-OpsQA-lyh
MODEL="${MODEL}" python /mnt/tenant-home_speed/lyh/opseval-configs/xz/generate.py
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python run.py /mnt/tenant-home_speed/lyh/opseval-configs/xz/runconfig.py -w /mnt/tenant-home_speed/lyh/outputs/xll/${MODEL}/ --mode infer -r