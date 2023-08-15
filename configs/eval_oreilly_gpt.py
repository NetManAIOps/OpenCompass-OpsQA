from mmengine.config import read_base
from opencompass.partitioners import SizePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    # Datasets
    from .datasets.oreilly.oreilly_test_gen import oreilly_datasets as oreilly
    # Models
    from .models.gpt_4_peiqi import models as gpt_4
    from .models.gpt_3dot5_turbo_peiqi import models as chatgpt_3dot5_turbo
    from .local_models.chatglm2_6b import models as chatglm2_6b
    from .local_models.internlm_chat_7b import models as internlm_chat_7b
    from .local_models.xverse_13b import models as xverse_13b
    from .local_models.qwen_chat_7b import models as qwen_chat_7b
    from .local_models.llama2_7b_chat import models as llama2_chat_7b
    from .models.hf_llama2_7b import models as hf_llama2_7b
    from .models.hf_chatglm2_6b import models as hf_chatglm2_6b
    from .models.hf_internlm_chat_7b import models as hf_internlm_chat_7b
    

datasets = [
    *oreilly
    ]

from opencompass.models import HuggingFaceCausalLM



models = [ 
    *gpt_4,                 # demo pass
    *chatgpt_3dot5_turbo,   # demo pass
    # *chatglm2_6b,           # demo pass
    # *internlm_chat_7b,      # demo pass?
    # *xverse_13b,            # demo pass?
    # *qwen_chat_7b,            
    # *llama2_chat_7b,
    # *hf_llama2_7b,
    # *hf_chatglm2_6b,
    # *hf_internlm_chat_7b,
        ]

# infer = dict(
#     partitioner=dict(
#         type=SizePartitioner,
#         max_task_size = 100,
#         gen_task_coef = 1,
#     ),
#     runner=dict(
#         type=LocalRunner,
#         max_num_workers=4,
#         task=dict(type=OpenICLInferTask),
#     ),
# )