from opencompass.models import Hisense
from mmengine import read_base

hisense = dict(abbr='hisense',
        type=Hisense,
        query_per_second=1, max_out_len=100, max_seq_len=2048, batch_size=1)