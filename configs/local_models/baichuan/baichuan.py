from opencompass.models import HuggingFaceCausalLM, BaichuanAPI

from mmengine import read_base
with read_base():
    from ...paths import ROOT_DIR
    from ..model_template import get_default_model
    from ...api_key import baichuan_key

_meta_template = dict(
    round=[
        dict(role="HUMAN", api_role="HUMAN"),
        dict(role="BOT", api_role="BOT", generate=True),
    ],
)

baichuan2_7b_base = get_default_model(abbr="baichuan2-7b", path=f"{ROOT_DIR}models/baichuan-inc/Baichuan2-7B-Base")
baichuan2_7b_chat = get_default_model(abbr="baichuan2-7b-chat", path=f"{ROOT_DIR}models/baichuan-inc/Baichuan2-7B-Chat", meta_template=_meta_template)

baichuan2_13b_base = get_default_model(abbr="baichuan2-13b", path=f"{ROOT_DIR}models/baichuan-inc/Baichuan2-13B-Base")
baichuan2_13b_chat = get_default_model(abbr="baichuan2-13b-chat", path=f"{ROOT_DIR}models/baichuan-inc/Baichuan2-13B-Chat", meta_template=_meta_template, num_gpus=1)

baichuan2_bases = [baichuan2_7b_base, baichuan2_13b_base]
baichuan2_chats = [baichuan2_7b_chat, baichuan2_13b_chat]


baichuan2_turbo = dict(abbr='Baichuan2-Turbo',
        type=BaichuanAPI, path='Baichuan2-Turbo', key=baichuan_key,
        max_out_len=100, max_seq_len=2048, batch_size=1)

baichuan3 = dict(abbr='Baichuan3',
        type=BaichuanAPI, path='Baichuan3', key=baichuan_key,
        max_out_len=100, max_seq_len=2048, batch_size=1)

baichuan_apis = [baichuan2_turbo, baichuan3]