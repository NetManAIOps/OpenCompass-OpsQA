from opencompass.models import OpenAIPeiqi
from mmengine import read_base
with read_base():
    from ...paths import ROOT_DIR
    from ...api_key import yyds_key

"""
claude-3-opus-20240229
claude-3-sonnet-20240229
claude-3-haiku-20240307
claude-2.1
claude-2.0
claude-instant-1.2
"""


claude_3_opus = dict(abbr='claude-3-opus',
        type=OpenAIPeiqi, 
        path='claude-3-opus-20240229',
        openai_api_base="https://ai-yyds.com/v1/chat/completions",
        key=yyds_key,
        query_per_second=1,
        max_out_len=100, max_seq_len=2048, batch_size=1)

claude_3_sonnet = dict(abbr='claude-3-sonnet',
        type=OpenAIPeiqi,
        path='claude-3-sonnet-20240229',
        openai_api_base="https://ai-yyds.com/v1/chat/completions",
        key=yyds_key,
        query_per_second=1,
        max_out_len=100, max_seq_len=2048, batch_size=1)

claude_3_haiku = dict(abbr='claude-3-haiku',
        type=OpenAIPeiqi,
        path='claude-3-haiku-20240307',
        openai_api_base="https://ai-yyds.com/v1/chat/completions",
        key=yyds_key,
        query_per_second=1,
        max_out_len=100, max_seq_len=2048, batch_size=1)
