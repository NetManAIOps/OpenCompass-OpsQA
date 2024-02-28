from .base import BaseModel, LMTemplateParser  # noqa
from .base_api import APITemplateParser, BaseAPIModel  # noqa
from .chatanywhere_api import OpenAIPeiqi  # noqa: F401, F403
from .baichuan_api import BaichuanAPI  # noqa: F401, F403
from .glm import GLM130B  # noqa: F401, F403
from .huggingface import HuggingFace  # noqa: F401, F403
from .huggingface import HuggingFaceCausalLM  # noqa: F401, F403
from .huggingface import QwenLM  # noqa: F401, F403
from .intern_model import InternLM  # noqa: F401, F403
from .llama2 import Llama2, Llama2Chat  # noqa: F401, F403
from .openai_api import OpenAI  # noqa: F401
# from .zhipuai_api import ZhiPuAI  # noqa
from .qwen_api import QwenAPI  # noqa
from .vllm_api import VllmAPI  # noqa
# from .custom import CustomModel  # noqa
from .wenxin_api import WenXinAI  # noqa
