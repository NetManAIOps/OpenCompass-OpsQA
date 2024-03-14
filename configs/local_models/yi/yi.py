from mmengine.config import read_base
with read_base():
    from ...paths import ROOT_DIR
    from ..model_template import get_default_model


yi_6b = get_default_model(abbr="yi-6b", path=f"{ROOT_DIR}models/01-ai/Yi-6B")
yi_9b = get_default_model(abbr="yi-9b", path=f"{ROOT_DIR}models/01-ai/Yi-9B")
yi_34b = get_default_model(abbr="yi-34b", path=f"{ROOT_DIR}models/01-ai/Yi-34B")

yi_6b_chat = get_default_model(abbr="yi-6b-chat", path=f"{ROOT_DIR}models/01-ai/Yi-6B-Chat")
yi_9b_chat = get_default_model(abbr="yi-9b-chat", path=f"{ROOT_DIR}models/01-ai/Yi-9B-Chat")
yi_34b_chat = get_default_model(abbr="yi-34b-chat", path=f"{ROOT_DIR}models/01-ai/Yi-34B-Chat")

yi_bases = [yi_6b, yi_9b, yi_34b]
yi_chats = [yi_6b_chat, yi_9b_chat, yi_34b_chat]
yi_all = yi_bases + yi_chats