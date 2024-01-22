from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union
import requests, json
from opencompass.registry import MODELS
from opencompass.utils import PromptList

from .base_api import BaseAPIModel

PromptType = Union[PromptList, str]


@MODELS.register_module()
class WenXinAI(BaseAPIModel):
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
        api_key: str,
        secret_key: str,
        path: str = 'ernie-bot-4.0',
        query_per_second: int = 1,
        max_seq_len: int = 2048,
        meta_template: Optional[Dict] = None,
        retry: int = 2,
        url: str = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro?access_token=", 
        temperature: Optional[float] = None
    ):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         query_per_second=query_per_second,
                         meta_template=meta_template,
                         retry=retry,)

        self.url = url
        self.access_token = self.get_access_token(api_key, secret_key)
        self.temperature = temperature

    def get_access_token(self, api_key, secret_key):
        """
        使用 API Key，Secret Key 获取access_token，替换下列示例中的应用API Key、应用Secret Key
        """
            
        url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={api_key}&client_secret={secret_key}"
        
        payload = json.dumps("")
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        response = requests.request("POST", url, headers=headers, data=payload)
        return response.json().get("access_token")

    def generate(
        self,
        inputs: List[str or PromptList],
        max_out_len: int = 512,
        temperature: Optional[float] = None
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
            # print(messages)
        #     # messages = messages[:-1]
        # print(messages)
        # print(len(messages))

        num_retries = 0
        while num_retries < self.retry:
            self.wait()
            try:
                payload = {'messages': messages}
                # print(payload)
                payload = json.dumps(payload)
                # print(payload)
                response = requests.request("POST", url=self.url + self.access_token, 
                                    headers={'Content-Type': 'application/json'}, 
                                    data=payload)
                if response.status_code != 200:
                    return ''
                if not response.json().get("result"):
                    if response.json().get("error_msg") and "max length" in response.json().get("error_msg"):
                        return response.json().get("error_msg")
                    print(response.json())
                    raise
                return response.json().get("result")
            except Exception as e:
                self.logger.error(e)
            num_retries += 1
        raise RuntimeError('Calling WenXin API failed after retrying for '
                           f'{self.retry} times. Check the logs for details.')
