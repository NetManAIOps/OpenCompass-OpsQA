from opencompass.models import HuggingFaceCausalLM

from mmengine.config import read_base
with read_base():
    from ...paths import ROOT_DIR
    from ..model_template import get_default_model

internlm2_chat_7b = dict(
        type=HuggingFaceCausalLM,
        abbr='internlm2_chat_7b',
        path=ROOT_DIR+"models/Shanghai_AI_Laboratory/internlm2-chat-7b",
        tokenizer_path=ROOT_DIR+'models/Shanghai_AI_Laboratory/internlm2-chat-7b',
        tokenizer_kwargs=dict(trust_remote_code=True),
        max_out_len=400,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )

internlm2_chat_20b = dict(
        type=HuggingFaceCausalLM,
        abbr='internlm2_chat_20b',
        path=ROOT_DIR+"models/Shanghai_AI_Laboratory/internlm2-chat-20b",
        tokenizer_path=ROOT_DIR+'models/Shanghai_AI_Laboratory/internlm2-chat-20b',
        tokenizer_kwargs=dict(trust_remote_code=True),
        max_out_len=400,
        max_seq_len=2048,
        batch_size=1,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=2, num_procs=1),
    )

internlm2_7b = get_default_model(abbr="internlm2-7b", path=ROOT_DIR+"models/Shanghai_AI_Laboratory/internlm2-7b")
internlm2_20b = get_default_model(abbr="internlm2-20b", path=ROOT_DIR+"models/Shanghai_AI_Laboratory/internlm2-20b", num_gpus=2)

internlm2_bases = [internlm2_7b, internlm2_20b]
internlm2_chats = [internlm2_chat_20b, internlm2_chat_7b]