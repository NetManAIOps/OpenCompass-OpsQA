from opencompass.models import BaichuanAPI
from mmengine.config import read_base


with read_base():
    from ...api_key import baichuan_key

baichuan2_turbo = dict(abbr='Baichuan2-Turbo',
        type=BaichuanAPI, path='Baichuan2-Turbo', key=baichuan_key,
        max_out_len=100, max_seq_len=2048, batch_size=1)

baichuan3 = dict(abbr='Baichuan3',
        type=BaichuanAPI, path='Baichuan3', key=baichuan_key,
        max_out_len=100, max_seq_len=2048, batch_size=1)