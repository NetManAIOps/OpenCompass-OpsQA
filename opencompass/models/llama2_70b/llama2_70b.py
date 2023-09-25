from typing import Dict, List, Optional, Union

import torch
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict

from opencompass.models.base import BaseModel
from opencompass.models.base_api import APITemplateParser
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList

from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

from .model import ModelArgs, Transformer
from .tokenizer import Tokenizer


PromptType = Union[PromptList, str]

def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

class Llama:
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        model_parallel_size: Optional[int] = None,
    ) -> "Llama":
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")
        if not model_parallel_is_initialized():
            if model_parallel_size is None:
                model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
            initialize_model_parallel(model_parallel_size)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        # seed must be the same in all processes
        torch.manual_seed(1)

        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        start_time = time.time()
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
        assert model_parallel_size == len(
            checkpoints
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"
        ckpt_path = checkpoints[get_model_parallel_rank()]
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params,
        )
        tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = tokenizer.n_words
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        model = Transformer(model_args)
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        return Llama(model, tokenizer)

    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        params = self.model.params
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        input_text_mask = tokens != pad_id
        for cur_pos in range(min_prompt_len, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            if logprobs:
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens[:, prev_pos + 1 : cur_pos + 1],
                    reduction="none",
                    ignore_index=pad_id,
                )
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                next_token == self.tokenizer.eos_id
            )
            prev_pos = cur_pos
            if all(eos_reached):
                break

        if logprobs:
            token_logprobs = token_logprobs.tolist()
        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            probs = None
            if logprobs:
                probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            if self.tokenizer.eos_id in toks:
                eos_idx = toks.index(self.tokenizer.eos_id)
                toks = toks[:eos_idx]
                probs = probs[:eos_idx] if logprobs else None
            out_tokens.append(toks)
            out_logprobs.append(probs)
        return (out_tokens, out_logprobs if logprobs else None)


class Llama2Chat(BaseModel):
    """LLaMA-2 chat model wrapper
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
        tokenizer_only: bool = False,
        tokenizer_path: Optional[str] = None,
        meta_template: Optional[Dict] = None,
        model_parallel_size: Optional[int] = None,
    ):  # noqa
        if tokenizer_only:
            self._load_tokenizer(tokenizer_path=tokenizer_path)
        else:
            self._load_model(path=path,
                             max_seq_len=max_seq_len,
                             max_batch_size=max_batch_size,
                             tokenizer_path=tokenizer_path,
                             model_parallel_size=model_parallel_size)
        self.max_seq_len = max_seq_len
        self.template_parser = APITemplateParser(meta_template)
        self.logger = get_logger()

    def _load_model(self,
                    path: str,
                    max_seq_len: int,
                    max_batch_size: int,
                    tokenizer_path: Optional[str] = None,
                    model_parallel_size: Optional[int] = None):
        self.generator = Llama.build(path, tokenizer_path, max_seq_len,
                                     max_batch_size, model_parallel_size)
        self.tokenizer = self.generator.tokenizer
        self.model = self.generator.model

    def _load_tokenizer(self, tokenizer_path: str):
        self.tokenizer = Tokenizer(model_path=tokenizer_path)

    def generate(self,
                 inputs: List[str or PromptList],
                 max_out_len: int = 512,
                 temperature: float = 0.6) -> str:
        """Generate response from input prompt.

        Args:
            inputs (list): input prompt
            max_out_len (int): max output length
            temperature (float): temperature for sampling
        """
        dialogs = []
        for input in inputs:
            assert isinstance(input, (str, PromptList))
            if isinstance(input, str):
                dialog = [{'role': 'user', 'content': input}]
            else:
                dialog = []
                for item in input:
                    msg = {'content': item['prompt']}
                    if item['role'] == 'HUMAN':
                        msg['role'] = 'user'
                    elif item['role'] == 'BOT':
                        msg['role'] = 'assistant'
                    elif item['role'] == 'SYSTEM':
                        msg['role'] = 'system'
                    dialog.append(msg)
            dialogs.append(dialog)

        try:
            results = self.generator.chat_completion(
                dialogs,  # type: ignore
                max_gen_len=max_out_len,
                temperature=temperature,
            )
            return [r['generation']['content'] for r in results]
        except AssertionError:
            self.warning('Batched data max token limit exceeded, '
                         'try to run one by one...')

        results = []
        for dialog in dialogs:
            try:
                result = self.generator.chat_completion(
                    [dialog],  # type: ignore
                    max_gen_len=max_out_len,
                    temperature=temperature,
                )[0]
                results.append(result['generation']['content'])
            except AssertionError:
                results.append('')
        return results

    def get_token_len(self, prompt: str) -> int:
        return len(self.tokenizer.encode(prompt, bos=True, eos=True)) + 100
