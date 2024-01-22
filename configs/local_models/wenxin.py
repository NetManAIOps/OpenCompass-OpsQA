from opencompass.models import WenXinAI
from mmengine.config import read_base

models = [
    dict(
        abbr='ernie-bot-4.0',
        type=WenXinAI, 
        api_key='82SG3HFTII1Uya2hyNCedCCo',
        secret_key='3FpsKdtGoOYvYpaDo76M4NSWXLKPGMrv',
        max_out_len=400, max_seq_len=2048, batch_size=1,
        temperature=0.7
        )
]
