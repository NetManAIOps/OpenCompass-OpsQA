from typing import Dict, List, Optional, Union
from copy import deepcopy
import torch
from concurrent.futures import ThreadPoolExecutor
from opencompass.models.base import BaseModel
from opencompass.models.base_api import APITemplateParser
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList
from typing import Dict, List, Optional, Tuple, Union
import chatglm_cpp

PromptType = Union[PromptList, str]


class CustomModel(BaseModel):
    """LLaMA-2 model wrapper
    https://github.com/facebookresearch/llama/tree/main.

    Args:
        path (str): path to the model directory
        max_seq_len (int): max sequence length
        max_batch_size (int): max batch size
        tokenizer_only (bool): whether to load tokenizer only
        tokenizer_path (str): path to the tokenizer directory
        meta_template (dict): meta template for the model
    """

    def __init__(
        self,
        path: str,
        max_seq_len: int = 2048,
        max_batch_size: int = 16,
        # tokenizer_only: bool = False,
        # tokenizer_path: Optional[str] = None,
        # meta_template: Optional[Dict] = None,
    ):  # noqa
        # if tokenizer_only:
        #     self._load_tokenizer(tokenizer_path=tokenizer_path)
        # else:
        #     self._load_model(path=path,
        #                      max_seq_len=max_seq_len,
        #                      max_batch_size=max_batch_size,
        #                      tokenizer_path=tokenizer_path)
        self.max_seq_len = max_seq_len
        # self.template_parser = APITemplateParser(meta_template)
        self._load_model(path=path)
        self.template_parser = LMTemplateParser(meta_template=None)
        self.logger = get_logger()

    def _load_model(self,
                    path: str, quanticize=False):
        
        self.model = chatglm_cpp.Pipeline(path)

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

    def _generate(self, inputs: str, max_out_len: int, temperature: Optional[float] = 0) -> List[str]:
        question = "你是一位运维专家,请简洁专业地回答下面的问题:\n" + inputs
        output_dict = self.model.chat([chatglm_cpp.ChatMessage(role="user",content=question)],
                                      top_p=0.7,top_k=10,
                                      temperature=0.2,
                                      repetition_penalty=1.1,
                                      stream=False)
        return output_dict.content
    
    

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


