from opencompass.models import VLLM

qwen_1_5b_14b_chat_vllm = dict(
    type=VLLM,
    abbr='qwen-1.5b-14b-chat',
    path="/home/junetheriver/models/qwen/Qwen1.5-14B-Chat",
    max_seq_len=2048,
    model_kwargs=dict(trust_remote_code=True, max_model_len=2048),
    generation_kwargs=dict(),
    meta_template=None,
    mode='none',
    batch_size=1,
    use_fastchat_template=False,
    end_str=None,
)