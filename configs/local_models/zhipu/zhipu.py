from opencompass.models import ZhiPuAI
from mmengine.config import read_base


with read_base():
    from ...api_key import zhipu_key

glm_4 = dict(
        abbr='zhipu_glm_4',
        type=ZhiPuAI, path='glm-4', key=zhipu_key,
        max_out_len=2048, max_seq_len=2048, batch_size=1,
        temperature=0.7
        )

glm_3_turbo = dict(
    abbr='zhipu_glm_3_turbo',
        type=ZhiPuAI, path='glm-3-turbo', key=zhipu_key,
        max_out_len=2048, max_seq_len=2048, batch_size=1,
        temperature=0.7
)
