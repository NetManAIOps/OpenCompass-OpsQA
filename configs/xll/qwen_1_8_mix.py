echo `date`, Setup the environment ...
set -e  # exit if error

CUDA=${1:-0,1,2,3,4,5,6,7}

cd /mnt/tenant-home_speed/lyh/OpenCompass-OpsQA-lyh
CUDA_VISIBLE_DEVICES=$CUDA python run.py configs/xll/qwen1_8B-mix5_1-sft_owl.py -w /mnt/tenant-home_speed/lyh/outputs/xll/qwen1_8B-mix5_1-sft_owl --mode infer -r
