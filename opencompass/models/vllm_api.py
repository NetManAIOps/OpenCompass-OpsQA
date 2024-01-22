from concurrent.futures import ThreadPoolExecutor
import re
import json
import requests
import threading
import warnings
from abc import abstractclassmethod
from copy import deepcopy
from time import sleep
from typing import Dict, List, Optional, Tuple, Union

from opencompass.utils import get_logger
from opencompass.utils.prompt import PromptList

PromptType = Union[PromptList, str]

from .base import BaseModel


class VllmAPI(BaseModel):
    """Base class for API model wrapper.

    Args:
        path (str): The path to the model.
        query_per_second (int): The maximum queries allowed per second
            between two consecutive calls of the API. Defaults to 1.
        retry (int): Number of retires if the API call fails. Defaults to 2.
        max_seq_len (int): The maximum sequence length of the model. Defaults
            to 2048.
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
    """

    is_api: bool = True

    def __init__(
        self,
        url: str = 'http://10.55.33.23:31066/generate',
        path: str = 'vllm_api',
        query_per_second: int = 4,
        max_seq_len: int = 512,
        meta_template: Optional[Dict] = None,
        stop = ['\n\n\n\n\n', '<|im_start|>','<|im_end|>', '<|endoftext|>', '</s>'],
        temperature: Optional[float] = 0,
        retry: int = 2,
    ):
        self.model = path
        self.url = url
        self.proxies = {'http':''}
        self.headers = {'accept': 'application/json',  'Content-Type': 'application/json'}
        # self.max_tokens = max_out_len
        self.temperature = temperature
        #self.stop = ['\n\n\n\n\n', '<|im_start|>','<|im_end|>', '<|endoftext|>', '</s>']
        self.stop = stop
        self.max_seq_len = max_seq_len
        self.meta_template = meta_template
        self.retry = retry
        self.query_per_second = query_per_second
        self.token_bucket = TokenBucket(query_per_second)
        self.template_parser = LMTemplateParser(meta_template)
        self.logger = get_logger()

    def generate(self, inputs: List[str], max_out_len: int, temperature: Optional[float] = 0,) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self._generate, inputs, [max_out_len] * len(inputs), [temperature] * len(inputs)))
        return results

    def _generate(
        self,
        input: str,
        max_out_len: int,
        temperature: Optional[float] = 0,
    ) -> str:
        """Generate results given an input.

        Args:
            inputs (str): A string.
            max_out_len (int): The maximum length of the output.

        Returns:
            str: The generated string.
        """
        messages = input
        num_retries = 0
        while num_retries < self.retry:
            self.wait()
            try:
                data = json.dumps({
                    "prompt": messages,
                    "max_tokens": max_out_len,
                    "temperature": self.temperature,
                    "stop": self.stop
                })
                response = requests.post(self.url, headers=self.headers, data=data, verify=False, proxies=self.proxies)
                return response.json()["text"][0][len(messages):].strip()
            except Exception as e:
                self.logger.error(e)
            num_retries += 1
        raise RuntimeError(f'Calling Vllm API failed after retrying for {self.retry} times. Check the logs for details.')

    @abstractclassmethod
    def get_ppl(self,
                inputs: List[str],
                mask_length: Optional[List[int]] = None) -> List[float]:
        """Get perplexity scores given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            mask_length (Optional[List[int]]): A list of mask lengths. If
                provided, the perplexity scores will be calculated with the
                first mask_length[i] tokens masked out. It's okay to skip
                its implementation if advanced features in PPLInfernecer is
                not needed.

        Returns:
            List[float]: A list of perplexity scores.
        """

    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized string. Only English and Chinese
        characters are counted for now. Users are encouraged to override this
        method if more accurate length is needed.

        Args:
            prompt (str): Input string.

        Returns:
            int: Length of the input tokens
        """

        english_parts = re.findall(r'[A-Za-z0-9]+', prompt)
        chinese_parts = re.findall(r'[\u4e00-\u9FFF]+', prompt)

        # Count English words
        english_count = sum(len(part.split()) for part in english_parts)

        # Count Chinese words
        chinese_count = sum(len(part) for part in chinese_parts)

        return english_count + chinese_count

    def wait(self):
        """Wait till the next query can be sent.

        Applicable in both single-thread and multi-thread environments.
        """
        return self.token_bucket.get_token()

    def to(self, device):
        pass


class LMTemplateParser:
    """Intermidate prompt template parser, specifically for language models.

    Args:
        meta_template (Dict): The meta template for the model.
    """

    def __init__(self, meta_template: Optional[Dict] = None):
        self.meta_template = meta_template
        if meta_template:
            assert 'round' in meta_template, 'round is required in meta' \
                ' template'
            assert isinstance(meta_template['round'], list)
            keys_to_check = ['round']

            if 'reserved_roles' in meta_template:
                assert isinstance(meta_template['reserved_roles'], list)
                keys_to_check.append('reserved_roles')

            self.roles: Dict[str, dict] = dict()  # maps role name to config
            for meta_key in keys_to_check:
                for item in meta_template[meta_key]:
                    assert isinstance(item, (str, dict))
                    if isinstance(item, dict):
                        assert item['role'] not in self.roles, \
                            'role in meta prompt must be unique!'
                        self.roles[item['role']] = item.copy()
                        # convert list of string and int into a raw string
                        # for the ease of future prompt processing
                        for key in ['begin', 'end']:
                            value = self.roles[item['role']].get(key, '')
                            if isinstance(value, list):
                                self.roles[item['role']][
                                    key] = self._encode_speical_tokens(value)

    def parse_template(self, prompt_template: PromptType, mode: str) -> str:
        """Parse a prompt template, and wrap it with meta template if
        applicable.

        Args:
            prompt_template (List[str or PromptList]): A prompt
                template (potentially before being wrapped by meta template).
            mode (str): Parsing mode. Choices are 'ppl' and 'gen'.

        Returns:
            str: The final string.
        """
        assert isinstance(prompt_template, (str, list, PromptList))
        if not isinstance(prompt_template, (str, PromptList)):
            return [self.parse_template(p, mode=mode) for p in prompt_template]

        assert mode in ['ppl', 'gen']
        if isinstance(prompt_template, str):
            return prompt_template
        if self.meta_template:

            prompt = ''
            # Whether to keep generating the prompt
            generate = True

            section_stack = []  # stores tuples: (section_name, start_idx)

            for i, item in enumerate(prompt_template):
                if not generate:
                    break
                if isinstance(item, str):
                    prompt += item
                elif isinstance(item, dict) and 'section' in item:
                    if item['pos'] == 'end':
                        section_name, start_idx = section_stack.pop(-1)
                        assert section_name == item['section']
                        if section_name in ['round', 'ice']:
                            dialogue = prompt_template[start_idx:i]
                            round_ranges = self._split_rounds(
                                dialogue, self.meta_template['round'])
                            # Consider inserting multiple round examples into
                            # template
                            for i in range(len(round_ranges) - 1):
                                start = round_ranges[i]
                                end = round_ranges[i + 1]
                                round_template = dialogue[start:end]
                                role_dict = self._update_role_dict(
                                    round_template)
                                new_str, generate = self._prompt2str(
                                    self.meta_template['round'],
                                    role_dict,
                                    # Start generating only when the mode is in
                                    # generation and the template reaches the
                                    # last round
                                    for_gen=mode == 'gen'
                                    and i == len(round_ranges) - 2
                                    and section_name == 'round')
                                prompt += new_str
                    elif item['pos'] == 'begin':
                        assert item['section'] in [
                            'begin', 'round', 'end', 'ice'
                        ]
                        section_stack.append((item['section'], i + 1))
                    else:
                        raise ValueError(f'Invalid pos {item["pos"]}')
                # if in "begin" or "end" section
                elif section_stack[-1][0] in ['begin', 'end']:
                    role_dict = self._update_role_dict(item)
                    new_str, generate = self._prompt2str(
                        item,
                        role_dict,
                        # never stop generation
                        for_gen=False)
                    prompt += new_str

            prompt = self.meta_template.get('begin', '') + prompt
            if generate:
                prompt += self.meta_template.get('end', '')

        else:
            # in case the model does not have any meta template
            prompt = ''
            last_sep = ''
            for item in prompt_template:
                if isinstance(item, dict) and set(['section', 'pos']) == set(
                        item.keys()):
                    continue
                if isinstance(item, str):
                    if item:
                        prompt += last_sep + item
                elif item.get('prompt', ''):  # it's a dict
                    prompt += last_sep + item.get('prompt', '')
                last_sep = '\n'
        return prompt

    def _split_rounds(
            self, prompt_template: List[Union[str, Dict]],
            single_round_template: List[Union[str, Dict]]) -> List[int]:
        """Split the prompt template into rounds, based on single round
        template.

        Return the index ranges of each round. Specifically,
        prompt_template[res[i]:res[i+1]] represents the i-th round in the
        template.
        """
        role_idxs = {
            role_cfg['role']: i
            for i, role_cfg in enumerate(single_round_template)
            if not isinstance(role_cfg, str)
        }
        last_role_idx = -1
        cutoff_idxs = [0]
        for idx, template in enumerate(prompt_template):
            if isinstance(template, str):
                continue
            role_idx = role_idxs[template['role']]
            if role_idx <= last_role_idx:
                cutoff_idxs.append(idx)
            last_role_idx = role_idx
        cutoff_idxs.append(len(prompt_template))
        return cutoff_idxs

    def _update_role_dict(self, prompt: Union[List, str,
                                              Dict]) -> Dict[str, Dict]:
        """Update the default role dict with the given prompt(s)."""
        assert isinstance(prompt, (str, list, dict))
        role_dict = deepcopy(self.roles)
        if isinstance(prompt, str):
            return role_dict
        if isinstance(prompt, dict):
            prompt = [prompt]
        for p in prompt:
            if isinstance(p, dict):
                role = p['role']
                if role not in self.roles:
                    role = p.get('fallback_role', None)
                    if not role:
                        print(f'{p} neither has an appropriate role nor '
                              'a fallback role.')
                role_dict[role].update(p)
        return role_dict

    def _prompt2str(self,
                    prompt: Union[List, str, Dict],
                    role_dict: Dict[str, Dict],
                    for_gen: bool = False) -> Tuple[str, bool]:
        """Convert the prompts to a string, given an updated role_dict.

        Args:
            prompts (Union[List, str, dict]): The prompt(s) to be converted.
            role_dict (Dict[str, Dict]): The updated role dict.
            for_gen (bool): If True, the prompts will be converted for
                generation tasks. The conversion stops before the first
                role whose "generate" is set to True.

        Returns:
            Tuple[str, bool]: The converted string, and whether the follow-up
            conversion should be proceeded.
        """
        assert isinstance(prompt, (list, str, dict))

        if isinstance(prompt, str):
            return prompt, True
        if isinstance(prompt, dict):
            return self._role2str(prompt, role_dict, for_gen)

        res = ''
        for p in prompt:
            new_str, cont = self._prompt2str(p, role_dict, for_gen)
            res += new_str
            if not cont:
                break
        return res, cont

    def _role2str(self,
                  role_prompt: Dict,
                  role_dict: Dict[str, Dict],
                  for_gen: bool = False) -> Tuple[str, bool]:
        """Convert a role prompt to a string, given an updated role_dict.

        Args:
            role_prompt (Dict): The role prompt to be converted.
            role_dict (Dict[str, Dict]): The updated role dict.
            for_gen (bool): If True, the prompts will be converted for
                generation tasks. The conversion stops before the first
                role whose "generate" is set to True.

        Returns:
            Tuple[str, bool]: The converted string, and whether the follow-up
            conversion should be proceeded.
        """
        merged_prompt = role_dict.get(
            role_prompt['role'],
            role_dict.get(role_prompt.get('fallback_role')))
        res = merged_prompt.get('begin', '')
        if for_gen and merged_prompt.get('generate', False):
            return res, False
        # res += merged_prompt.get('prompt', '') + merged_prompt.get('end', '')
        res += merged_prompt.get('prompt', '') + merged_prompt.get('end', '')
        return res, True

    def _encode_speical_tokens(self, prompt: List[Union[str, int]]) -> str:
        """Encode the special tokens in the prompt.

        Now this is left for the future work
        """
        raise NotImplementedError('Using List[str|int] is as the begin or end'
                                  'of a prompt is not supported yet.')
        res = ''
        for item in prompt:
            if isinstance(item, str):
                res += item
            else:
                res += f'<META_TOKEN_{item}>'
        return res

class TokenBucket:
    """A token bucket for rate limiting.

    Args:
        query_per_second (float): The rate of the token bucket.
    """

    def __init__(self, rate):
        self._rate = rate
        self._tokens = threading.Semaphore(0)
        self.started = False

    def _add_tokens(self):
        """Add tokens to the bucket."""
        while True:
            if self._tokens._value < self._rate:
                self._tokens.release()
            sleep(1 / self._rate)

    def get_token(self):
        """Get a token from the bucket."""
        if not self.started:
            self.started = True
            threading.Thread(target=self._add_tokens, daemon=True).start()
        self._tokens.acquire()
