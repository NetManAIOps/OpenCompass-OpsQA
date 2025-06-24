# -*- coding: utf-8 -*-
# @Time =2024/11/14
# @Author = wanglisen
# @Email = wanglisen@chinamobile.com
# @Project = code

import jwt
import time
import json
import requests
from requests.exceptions import RequestException
from typing import Dict, List, Optional, Union
import jieba
import re
from loguru import logger
from opencompass.registry import MODELS
from opencompass.utils.prompt import PromptList
from concurrent.futures import ThreadPoolExecutor
from .base_api import BaseAPIModel

PromptType = Union[PromptList, str]


@MODELS.register_module()
class CHMobile(BaseAPIModel):

    is_api: bool = True

    def __init__(self,
                 max_seq_len: int = 4096,
                 query_per_second: int = 1,
                 rpm_verbose: bool = False,
                 retry: int = 2,
                 meta_template: Optional[Dict] = None,
                 mode: str = 'none',
                 temperature: Optional[float] = 0.7):

        super().__init__(path="chmobile",
                         max_seq_len=max_seq_len,
                         meta_template=meta_template,
                         query_per_second=query_per_second,
                         rpm_verbose=rpm_verbose,
                         retry=retry)
        
        import tiktoken
        self.tiktoken = tiktoken
        try:
            enc = self.tiktoken.encoding_for_model("hisense")
            self.tiktoken_model = "hisense"
        except KeyError:
            self.tiktoken_model = 'gpt-4'

        self.mode = mode
        self.temperature = temperature
        self.max_seq_len = max_seq_len
        #self.url = 'https://jiutian.10086.cn/kunlun/ingress/api/h3t-f9c8f9/1151629e072a417ea9fda5a067d2891f/ai-c97aea0d99d04731b33a11b97a3a4ef4/service-aa5423e79c1f4054b888d1127e4884c5/generate'
        self.url = "https://jiutian.10086.cn/kunlun/ingress/api/h3t-dfac2f/1151629e072a417ea9fda5a067d2891f/ai-baee6f30e28247408fb87f66918d547f/service-298efc68a30f4e8ea447e3395dfa3ab4/generate"
        self.token = self.get_kc_act()


    def get_kc_act(self):
        url = 'https://jiutian.10086.cn/auth/realms/TechnicalMiddlePlatform/protocol/openid-connect/token'
        user = 'wanglisen'
        password = 'wWw118118'
        client_id = 'kunlun-front'
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded'
        }

        payload = 'grant_type=' + 'password&username=' + user + '&password=' + password + '&scope=openid&client_id=' + client_id
        response = requests.request("POST", url, headers=headers, data=payload)
        print(response.json())
        access_token = response.json()["access_token"]
        auth = access_token
        return auth


    def check_token_exp(self, access_token):
        try:
            res = jwt.decode(access_token, key='kunlun-front', algorithms=['HS256'], options={"verify_signature": False})
            exp = int(res["exp"])
            current_time = int(time.time())
            return current_time > exp
        except jwt.ExpiredSignatureError:
            return True
        except Exception as e:
            print(f"Error checking token expiration: {e}")
            return True


    def get_header(self):
        access_status = self.check_token_exp(self.token)
        if access_status:
            # logger.debug("token过期,重新获取")
            time.sleep(60)
            self.token = self.get_kc_act()
            headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.token}
        else:
            # logger.debug("token未过期")
            headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.token}
        return headers


    def retry_on_no_response(max_retries=3, backoff_factor=1):
        def decorator(func):
            def wrapper(*args, **kwargs):
                retries = 0
                while retries < max_retries:
                    try:
                        response = func(*args, **kwargs)
                        if response.status_code == 200:
                            return response
                        else:
                            print(f"Received non-200 response: {response.status_code}")
                    except RequestException as e:
                        print(f"An error occurred: {e}")
                        print(f"response: {response}")
                    retries += 1
                    time.sleep(backoff_factor * (2 ** retries))  # Exponential backoff
                print("Max retries exceeded with no success.")
                return None

            return wrapper
        return decorator


    # @retry_on_no_response(max_retries=2)
    def model_request(self, prompt, URL=''):
        # vllm payload
        prompt = "Human:\n" + prompt + "\n\nAssistant:\n"

        payload = json.dumps({
            "inputs": prompt,
            "parameters": {
                "do_sample": False,
                "max_new_tokens": self.max_seq_len
                }
        })

        headers = self.get_header()

        try:
            response = None
            response = requests.request("POST", URL, headers=headers, data=payload,)
            #  timeout=900, stream=False)
            # print('原始response:\n', response)

            answer = response.json()

            result = answer["generated_text"]
        except Exception as e:
            print(f"An error occurred: {e}")
            print(f"response: {response}")
            return ""

        # print(result)
        return result
        res=find_first_appearance(result)
        return res

    def generate(
        self,
        inputs: List[str or PromptList],
        max_out_len: int = 512,
        temperature: float = 0.7,
    ) -> List[str]:
        with ThreadPoolExecutor() as executor:
            results = list(
                executor.map(self._generate, inputs,
                             [max_out_len] * len(inputs),
                             [temperature] * len(inputs)))
        return results
    
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
    
    def _generate(self, input: str or PromptList, max_out_len: int,
                  temperature: float) -> str:
        if isinstance(input, str) and self.mode != 'none':
            context_window = self.max_seq_len
            input = self.bin_trim(input, context_window - 100 - max_out_len)

        assert isinstance(input, str)

        return self.model_request(input, self.url)

def find_first_appearance(text):
    # 定义字母选项
    options = ['A', 'B', 'C', 'D']

    # 记录最早出现的字母及其位置
    first_letter = None
    first_position = len(text)

    # 遍历选项，找到每个字母在文本中第一次出现的位置
    for option in options:
        position = text.find(option)
        if position != -1 and position < first_position:
            first_position = position
            first_letter = option

    return first_letter if first_letter else "None of A, B, C, D found"

prompt="以下是中国关于无线网考试的单项选择题，请选出其中的正确答案。\n CU/DU合一情况下，数据传输不包括（    ）。\n A. 中传 \n B. 前传 \n C. 回传 \n D. 远传 \n 答案：A \n CUPS架构引入新接口，其控制平面上运行的是什么协议（    ）。\n A. HTTP \n B. PFCP \n C. NG-AP \n D. Diameter \n 答案：B \n dBi和dBd是功率增益的单位，两者都是相对值，但参考基准不一样，其中dBi的参考基准（    ）。 \n A. 全向天线 \n B. 偶极子 \n C. 定向天线 \n D. 非定向天线 \n 答案：A \n C波段与毫米波在视距传播时，覆盖上的差距大约是多少？ \n A. 10dB \n B. 20dB \n C. 50dB \n D. 80dB \n 答案：B \n eLTE eNB 和gNB之间的接口称为（    ）接口\n A. X1 \n B. X2 \n C. Xn \n D. Xx \n 答案：C \n 通信电池(2V系列)通常浮充使用放电深度超过80%时, 设计循环使用寿命次数（   ）。\n A. 循环使用寿命大于1200次 \n B. 循环使用寿命大于800次 \n C. 循环使用寿命大于300次 \n D. 循环使用寿命大于100次 \n  答案："

if __name__ == '__main__':
    model=CHMobile()
    answer=model.generate(prompt)
    print(answer)


