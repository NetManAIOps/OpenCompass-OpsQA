echo `date`, Setup the environment ...
set -e  # exit if error

cd /mnt/tenant-home_speed/lyh/OpenCompass-OpsQA-lyh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run.py /mnt/tenant-home_speed/lyh/opseval-configs/zzr/runconfig.py -w /mnt/tenant-home_speed/lyh/outputs/xll/nm_qwen_14b_3gpp_mix_sft_owlb_ppl/  -r

echo `date`, Setup the environment ...
set -e  # exit if error
cd /mnt/tenant-home_speed/lyh/OpenCompass-OpsQA-lyh
CUDA_VISIBLE_DEVICES=6,7 python run.py /mnt/tenant-home_speed/lyh/opseval-configs/zzr/runconfig.py -w /mnt/tenant-home_speed/lyh/outputs/xll/qwen14b_chat_ptfull_3gpp_sft_sharegpt/ -r