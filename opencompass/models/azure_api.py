import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Dict, List, Optional, Union
from langchain_openai.chat_models import AzureChatOpenAI

import jieba
import requests

from opencompass.registry import MODELS
from opencompass.utils.prompt import PromptList

from .base_api import BaseAPIModel

PromptType = Union[PromptList, str]

@MODELS.register_module()
class OpenAIAzure(BaseAPIModel):
    """Model wrapper around OpenAI's models.

    Args:
        path (str): The name of OpenAI's model.
        max_seq_len (int): The maximum allowed sequence length of a model.
            Note that the length of prompt + generated tokens shall not exceed
            this value. Defaults to 2048.
        query_per_second (int): The maximum queries allowed per second
            between two consecutive calls of the API. Defaults to 1.
        retry (int): Number of retires if the API call fails. Defaults to 2.
        key (str or List[str]): OpenAI key(s). In particular, when it
            is set to "ENV", the key will be fetched from the environment
            variable $OPENAI_API_KEY, as how openai defaults to be. If it's a
            list, the keys will be used in round-robin manner. Defaults to
            'ENV'.
        org (str or List[str], optional): OpenAI organization(s). If not
            specified, OpenAI uses the default organization bound to each API
            key. If specified, the orgs will be posted with each request in
            round-robin manner. Defaults to None.
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        openai_api_base (str): The base url of OpenAI's API. Defaults to
            'https://api.openai.com/v1/chat/completions'.
        mode (str, optional): The method of input truncation when input length
            exceeds max_seq_len. 'front','mid' and 'rear' represents the part
            of input to truncate. Defaults to 'none'.
        temperature (float, optional): What sampling temperature to use.
            If not None, will override the temperature in the `generate()`
            call. Defaults to None.
    """

    is_api: bool = True

    def __init__(self,
                 endpoint: str,
                 api_key: str,
                 deployment: str = 'gpt4o-test',
                 api_version = "2024-02-01",
                 max_seq_len: int = 4096,
                 query_per_second: int = 1,
                 rpm_verbose: bool = False,
                 retry: int = 2,
                 meta_template: Optional[Dict] = None,
                 mode: str = 'none',
                 temperature: Optional[float] = 0.7):

        super().__init__(path=deployment,
                         max_seq_len=max_seq_len,
                         meta_template=meta_template,
                         query_per_second=query_per_second,
                         rpm_verbose=rpm_verbose,
                         retry=retry)
        import tiktoken
        self.tiktoken = tiktoken
        try:
            enc = self.tiktoken.encoding_for_model(deployment)
            self.tiktoken_model = deployment
        except KeyError:
            self.tiktoken_model = 'gpt-4'

        self.temperature = temperature
        self.mode = mode
        self.api_version = api_version
        self.api_key = api_key
        self.deployment = deployment

        self.endpoint = endpoint

        self.model = AzureChatOpenAI(
            api_key=self.api_key,
            openai_api_version=self.api_version,
            azure_endpoint=self.endpoint,
            azure_deployment=self.deployment,
            temperature=temperature,
            model_kwargs={"top_p": 0.95},
        )

    def generate(
        self,
        inputs: List[str or PromptList],
        max_out_len: int = 512,
        temperature: float = 0.7,
    ) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[str or PromptList]): A list of strings or PromptDicts.
                The PromptDict should be organized in OpenCompass'
                API format.
            max_out_len (int): The maximum length of the output.
            temperature (float): What sampling temperature to use,
                between 0 and 2. Higher values like 0.8 will make the output
                more random, while lower values like 0.2 will make it more
                focused and deterministic. Defaults to 0.7.

        Returns:
            List[str]: A list of generated strings.
        """
        if self.temperature is not None:
            temperature = self.temperature

        with ThreadPoolExecutor() as executor:
            results = list(
                executor.map(self._generate, inputs,
                             [max_out_len] * len(inputs),
                             [temperature] * len(inputs)))
        return results

    def _generate(self, input: str or PromptList, max_out_len: int,
                  temperature: float) -> str:
        """Generate results given a list of inputs.

        Args:
            inputs (str or PromptList): A string or PromptDict.
                The PromptDict should be organized in OpenCompass'
                API format.
            max_out_len (int): The maximum length of the output.
            temperature (float): What sampling temperature to use,
                between 0 and 2. Higher values like 0.8 will make the output
                more random, while lower values like 0.2 will make it more
                focused and deterministic.

        Returns:
            str: The generated string.
        """
        assert isinstance(input, (str, PromptList))

        # max num token for gpt-3.5-turbo is 4097
        context_window = 4096
        if '32k' in self.path:
            context_window = 32768
        elif '16k' in self.path:
            context_window = 16384
        elif 'gpt-4' in self.path:
            context_window = 8192

        # will leave 100 tokens as prompt buffer, triggered if input is str
        if isinstance(input, str) and self.mode != 'none':
            context_window = self.max_seq_len
            input = self.bin_trim(input, context_window - 100 - max_out_len)

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
                elif item['role'] == 'SYSTEM':
                    msg['role'] = 'system'
                messages.append(msg)

        # Hold out 100 tokens due to potential errors in tiktoken calculation
        max_out_len = min(
            max_out_len, context_window - self.get_token_len(str(input)) - 100)
        if max_out_len <= 0:
            return ''

        max_num_retries = 0
        while max_num_retries < self.retry:
            self.wait()

            try:
                # data = dict(
                #     model=self.path,
                #     messages=messages,
                #     max_tokens=max_out_len,
                #     n=1,
                #     stop=None,
                #     temperature=temperature,
                # )
                # raw_response = requests.post(self.url,
                #                              headers=header,
                #                              data=json.dumps(data))
                raw_response = self.model.invoke(messages)

                return raw_response.content

            except requests.ConnectionError:
                self.logger.error('Got connection error, retrying...')
                max_num_retries += 1
                continue

        raise RuntimeError('Calling OpenAI failed after retrying for '
                           f'{max_num_retries} times. Check the logs for '
                           'details.')

    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized string. Only English and Chinese
        characters are counted for now. Users are encouraged to override this
        method if more accurate length is needed.

        Args:
            prompt (str): Input string.

        Returns:
            int: Length of the input tokens
        """
        enc = self.tiktoken.encoding_for_model(self.tiktoken_model)
        return len(enc.encode(prompt))

    def bin_trim(self, prompt: str, num_token: int) -> str:
        """Get a suffix of prompt which is no longer than num_token tokens.

        Args:
            prompt (str): Input string.
            num_token (int): The upper bound of token numbers.

        Returns:
            str: The trimmed prompt.
        """
        token_len = self.get_token_len(prompt)
        if token_len <= num_token:
            return prompt
        pattern = re.compile(r'[\u4e00-\u9fa5]')
        if pattern.search(prompt):
            words = list(jieba.cut(prompt, cut_all=False))
            sep = ''
        else:
            words = prompt.split(' ')
            sep = ' '

        l, r = 1, len(words)
        while l + 2 < r:
            mid = (l + r) // 2
            if self.mode == 'front':
                cur_prompt = sep.join(words[-mid:])
            elif self.mode == 'mid':
                cur_prompt = sep.join(words[:mid]) + sep.join(words[-mid:])
            elif self.mode == 'rear':
                cur_prompt = sep.join(words[:mid])

            if self.get_token_len(cur_prompt) <= num_token:
                l = mid  # noqa: E741
            else:
                r = mid

        if self.mode == 'front':
            prompt = sep.join(words[-l:])
        elif self.mode == 'mid':
            prompt = sep.join(words[:l]) + sep.join(words[-l:])
        elif self.mode == 'rear':
            prompt = sep.join(words[:l])
        return prompt