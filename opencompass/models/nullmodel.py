from typing import Dict, List, Optional, Union
from opencompass.registry import MODELS
from opencompass.utils.prompt import PromptList
from .base_api import BaseAPIModel

PromptType = Union[PromptList, str]


@MODELS.register_module()
class NullModel(BaseAPIModel):
    def generate(self, inputs: List[str or PromptList], max_out_len: int = 512, temperature: float = 0.7) -> List[str]:
        return [""] * len(inputs)