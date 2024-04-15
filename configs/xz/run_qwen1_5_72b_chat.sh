#!/bin/sh
set -x
set -e

MODEL_PATH="/mnt/tenant-home_speed/gaozhengwei/projects/LLM/models/Qwen/Qwen1.5-72B-Chat"
PORT=12310
GPUS=("0,1" "2,3" "4,5" "6,7")

source /root/miniconda3/etc/profile.d/conda.sh && conda activate vllm
for i in {0..3}; do
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} ray start --head --port $((8012 + $i)) --num-cpus 2
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} ray start --address=localhost:$((8012 + $i)) --num-cpus 2
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} RAY_ADDRESS=localhost:$((8012 + $i)) python -m vllm.entrypoints.openai.api_server \
        --model $MODEL_PATH --host 127.0.0.1 --port $(($PORT + $i)) --tensor-parallel-size 2 --gpu-memory-utilization 0.98 --trust-remote-code --max-model-len 2048 & pid[$i]=$!
    echo "port=$(($PORT + $i)), pid=${pid[$i]}"
done
echo "[VLLM] All backend servers have been started!!!"

wait
echo "[VLLM] All backend services have been successfully killed!!!"
ray stop
echo "[VLLM] Ray stoped"
