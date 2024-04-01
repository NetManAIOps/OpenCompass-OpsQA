from mmengine.config import read_base
with read_base():
    from ...paths import ROOT_DIR
    from ..model_template import get_default_model, get_vllm_model

yi_meta_template = dict(
    round=[
        dict(role="HUMAN", begin='<|im_start|>user\n', end='<|im_end|>'),
        dict(role="BOT", begin="<|im_start|>assistant\n", end='<|im_end|>', generate=True),
    ],
)

yi_6b = get_default_model(abbr="yi-6b", path=f"{ROOT_DIR}models/01-ai/Yi-6B")
yi_9b = get_default_model(abbr="yi-9b", path=f"{ROOT_DIR}models/01-ai/Yi-9B")
yi_34b = get_default_model(abbr="yi-34b", path=f"{ROOT_DIR}models/01-ai/Yi-34B", num_gpus=2)

yi_6b_chat = get_default_model(abbr="yi-6b-chat", path=f"{ROOT_DIR}models/01-ai/Yi-6B-Chat", meta_template=yi_meta_template)
# yi_9b_chat = get_default_model(abbr="yi-9b-chat", path=f"{ROOT_DIR}models/01-ai/Yi-9B-Chat", meta_template=yi_meta_template)
yi_34b_chat = get_default_model(abbr="yi-34b-chat", path=f"{ROOT_DIR}models/01-ai/Yi-34B-Chat", meta_template=yi_meta_template, num_gpus=4)

yi_bases = [yi_6b, yi_9b, yi_34b]
yi_chats = [yi_6b_chat, 
            # yi_9b_chat, 
            yi_34b_chat]
yi_all = yi_bases + yi_chats




yi_6b_vllm = get_vllm_model(abbr="yi-6b", path=f"{ROOT_DIR}models/01-ai/Yi-6B")
yi_9b_vllm = get_vllm_model(abbr="yi-9b", path=f"{ROOT_DIR}models/01-ai/Yi-9B")
yi_34b_vllm = get_vllm_model(abbr="yi-34b", path=f"{ROOT_DIR}models/01-ai/Yi-34B", num_gpus=2)

yi_6b_chat_vllm = get_vllm_model(abbr="yi-6b-chat", path=f"{ROOT_DIR}models/01-ai/Yi-6B-Chat", meta_template=yi_meta_template)
# yi_9b_chat_vllm = get_vllm_model(abbr="yi-9b-chat", path=f"{ROOT_DIR}models/01-ai/Yi-9B-Chat", meta_template=yi_meta_template)
yi_34b_chat_vllm = get_vllm_model(abbr="yi-34b-chat", path=f"{ROOT_DIR}models/01-ai/Yi-34B-Chat", meta_template=yi_meta_template, num_gpus=4)

yi_bases_vllm = [yi_6b_vllm, yi_9b_vllm, yi_34b_vllm]
yi_chats_vllm = [yi_6b_chat_vllm, 
                #  yi_9b_chat_vllm, 
                 yi_34b_chat_vllm]
yi_all_vllm = yi_bases_vllm + yi_chats_vllm