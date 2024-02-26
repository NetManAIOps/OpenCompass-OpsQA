from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union
from zhipuai import ZhipuAI
from opencompass.registry import MODELS
from opencompass.utils import PromptList

from .base_api import BaseAPIModel

PromptType = Union[PromptList, str]


@MODELS.register_module()
class ZhiPuAI(BaseAPIModel):
    """Model wrapper around Claude API.

    Args:
        key (str): Authorization key.
        path (str): The model to be used. Defaults to claude-2.
        query_per_second (int): The maximum queries allowed per second
            between two consecutive calls of the API. Defaults to 1.
        max_seq_len (int): Unused here.
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        retry (int): Number of retires if the API call fails. Defaults to 2.
    """

    def __init__(
        self,
        key: str,
        path: str = 'chatglm_pro',
        query_per_second: int = 2,
        max_seq_len: int = 2048,
        meta_template: Optional[Dict] = None,
        retry: int = 2,
        temperature: Optional[float] = None
    ):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         query_per_second=query_per_second,
                         meta_template=meta_template,
                         retry=retry,)
        try:
            import zhipuai
        except ImportError:
            raise ImportError('Import zhipuai failed. Please install it '
                              'with "pip install zhipuai" and try again.')

        self.client = ZhipuAI(api_key=key)
        # zhipuai.api_key = key
        self.model = path
        # self.zhipuai = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        self.temperature = temperature

    def generate(
        self,
        inputs: List[str or PromptList],
        max_out_len: int = 512,
        temperature: float = 0,
    ) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[str or PromptList]): A list of strings or PromptDicts.
                The PromptDict should be organized in OpenCompass'
                API format.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        with ThreadPoolExecutor() as executor:
            results = list(
                executor.map(self._generate, inputs))
        return results

    def _generate(
        self,
        input: str or PromptList,
    ) -> str:
        """Generate results given an input.

        Args:
            inputs (str or PromptList): A string or PromptDict.
                The PromptDict should be organized in OpenCompass'
                API format.
            max_out_len (int): The maximum length of the output.

        Returns:
            str: The generated string.
        """
        if isinstance(input, str):
            messages = [{'role': 'user', 'content': input}]
        else:
            messages = []
            for item in input:
                msg = {'content': item['prompt']}
                if item['role'] == 'HUMAN':
                    msg['role'] = 'user'
                elif item['role'] == 'BOT':
                    msg['role'] = 'assistant'
                messages.append(msg)

        num_retries = 0
        while num_retries < self.retry:
            self.wait()
            try:
                # response = self.zhipuai.invoke(
                #     model=self.model,
                #     prompt=messages
                # )
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                )
                # if response['code'] != 200:
                #     return ''
                return response.choices[0].message.content
            except Exception as e:
                self.logger.error(e)
            num_retries += 1
        raise RuntimeError('Calling ZhiPu API failed after retrying for '
                           f'{self.retry} times. Check the logs for details.')
