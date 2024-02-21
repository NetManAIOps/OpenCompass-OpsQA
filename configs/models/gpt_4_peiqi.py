from opencompass.models import OpenAIPeiqi
from mmengine.config import read_base


with read_base():
    from ..api_key import peiqi_key

models = [
    dict(
        abbr='GPT-4-peiqi',
        type=OpenAIPeiqi, path='gpt-4-0613', key=peiqi_key,
        max_out_len=2048, max_seq_len=2048, batch_size=1,
        retry=5,
        temperature=0
        )
]
