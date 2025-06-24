from opencompass.models import NullModel
from mmengine.config import read_base

models = [
    dict(
        abbr='nullmodel',
        type=NullModel, 
        path='nullmodel',
        batch_size=1,
        # max_out_len=400, max_seq_len=2048, batch_size=1,
        # # temperature=0.7
        )
]
