from opencompass.models import ZhiPuAI
from mmengine.config import read_base


with read_base():
    from ..api_key import zhipu_key

models = [
    dict(
        abbr='zhipu_chatglm_pro',
        type=ZhiPuAI, path='chatglm_pro', key=zhipu_key,
        max_out_len=2048, max_seq_len=2048, batch_size=1,
        temperature=0.7
        )
]
