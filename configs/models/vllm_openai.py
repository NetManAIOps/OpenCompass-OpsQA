from opencompass.models import OpenAIPeiqi
from mmengine.config import read_base

models = [
    dict(abbr='Qwen1.5-32B-Chat',
        type=OpenAIPeiqi, 
        path="/home/junetheriver/models/qwen/Qwen1.5-32B-Chat", 
        openai_api_base="http://101.6.5.144:20242/v1/chat/completions",
        key="EMPTY",
        max_out_len=100, max_seq_len=4096, batch_size=1)
]
