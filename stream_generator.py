"""
https://github.com/Lynn1  2024.5.11 update:
Add stream generation support functions
"""

from typing import List, Optional
from llama3.llama import Dialog, Llama
import torch
import torch.nn.functional as F
from llama3.llama.generation import sample_top_p
from typing import List, Optional, Tuple, TypedDict


class LLMGenerator:
    def __init__(self,
                 ckpt_dir: str,
                 tokenizer_path: str,
                 temperature: float = 0.6,
                 top_p: float = 0.9,
                 max_seq_len: int = 512,
                 max_batch_size: int = 4,
                 max_gen_len: Optional[int] = None):
                  

        self.temperature = temperature
        self.top_p = top_p
        self.max_gen_len = max_gen_len

        self.generator = Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
        )

        self.dialogs: List[Dialog] = [
        [{"role": "system", "content": "你是一个聊天助手，请始终用中文回答问题。"}] # You can change different system prompts here
        ]

    @torch.inference_mode()
    def stream_generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False):

        params = self.generator.model.params
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len, f"{max_prompt_len} <= {params.max_seq_len}"
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.generator.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        input_text_mask = tokens != pad_id
        if min_prompt_len == total_len:
            logits = self.generator.model.forward(tokens, prev_pos)
            token_logprobs = -F.cross_entropy(
                input=logits.transpose(1, 2),
                target=tokens,
                reduction="none",
                ignore_index=pad_id,
            )

        stop_tokens = torch.tensor(list(self.generator.tokenizer.stop_tokens))
        
        for cur_pos in range(min_prompt_len, total_len):
            logits = self.generator.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_yields = next_token.tolist() 

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            if logprobs:
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens[:, prev_pos + 1 : cur_pos + 1],
                    reduction="none",
                    ignore_index=pad_id,
                )
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                torch.isin(next_token, stop_tokens)
            )

            next_logprobs = []
            if logprobs:
                next_logprobs = token_logprobs[:, prev_pos + 1 : cur_pos + 1].tolist()

            prev_pos = cur_pos
            if all(eos_reached):
                break

            yield (next_yields, next_logprobs if logprobs else None)

    
    def stream_chat(self, user_query):
        """
        Stream a response to each request from the user.
        """
        self.dialogs[0].append({"role": "user", "content": user_query})

        if self.max_gen_len is None:
            self.max_gen_len = self.generator.model.params.max_seq_len - 1

        prompt_tokens = [
            self.generator.formatter.encode_dialog_prompt(self.dialogs[0])
        ]
        # generate next word:
        res = ""
        buffer = []
        for next_yields, next_logprobs in self.stream_generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=self.max_gen_len,
            temperature=self.temperature,
            top_p=self.top_p,
            logprobs=False,
        ):
            try: #fix the tokenizer decode error by using decode with errors="strict"
                buffer.extend(next_yields[0])
                next_word = self.generator.tokenizer.model.decode(buffer,errors="strict")
                # next_word = self.generator.tokenizer.decode(buffer)
                res += next_word
                buffer = [] # clear the buffer
                yield next_word
            except UnicodeDecodeError:
                continue
        
        # save the old dialog
        self.dialogs[0].append({"role": "assistant", "content": res})
        if len(self.dialogs[0]) > 7: # limit the saved old dialog length
            self.dialogs[0] = self.dialogs[0][:3] + self.dialogs[0][-4:]
        
        yield "<user_end>"
