from opencompass.models import OpenAIPeiqi
from mmengine.config import read_base


with read_base():
    from ..api_key import peiqi_key

models = [
    dict(abbr='GPT-3.5-turbo',
        type=OpenAIPeiqi, path='gpt-3.5-turbo', key=peiqi_key,
        max_out_len=100, max_seq_len=2048, batch_size=1)
]
