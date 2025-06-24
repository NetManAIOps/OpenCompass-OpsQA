from opencompass.models import CHMobile
from mmengine import read_base

chmobile = dict(abbr='chmobile',
        type=CHMobile,
        query_per_second=1, max_out_len=200, max_seq_len=200, batch_size=1)