import torch
from typing import List, Dict, Optional, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from dataclasses import dataclass

from edlang.entrypoints.config import EngineConfig


@dataclass
class Request:
    request_id: int
    prompt: str
    max_new_tokens: int
    current_len: int = 0
    sampling_params: Optional[Dict[str, Any]] = None  # Bonus Part

    input_ids: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    past_key_values: Optional[Any] = None
    generated_tokens: Optional[List[int]] = None
    generated_text: Optional[str] = None
    num_generated: int = 0
    is_finished: bool = False


@dataclass
class BatchResult:
    request_ids: List[int]
    new_tokens: List[List[int]]
    finished: List[bool]


class InferenceEngine:
    def __init__(self, engine_config: EngineConfig):
        self.model_config = engine_config.model_config

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "right"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_name,
            torch_dtype=self.model_config.torch_dtype,
            device_map=self.model_config.device,
        )
        self.model.eval()

    @torch.no_grad()
    def prefill(self, requests: List[Request]) -> BatchResult:
        """
        Prefill phase: tokenize prompts, run through model, generate first token.
        
        Steps:
        1. Tokenize prompts and create batch
        2. Forward pass with use_cache=True to get logits and KV cache
        3. Generate first token for each request (greedy: argmax)
        4. Save request state (input_ids, attention_mask, past_key_values)
        5. Check if finished (EOS token or max_new_tokens reached)
        
        Note: Use attention_mask to get real prompt length (without padding).
        """
        if not requests:
            return BatchResult(request_ids=[], new_tokens=[], finished=[])

        # TODO: Tokenize prompts and create batch (use self.tokenizer with padding=True)
        # TODO: Forward pass through model
        # TODO: For each request:
        #   - Get real prompt length from attention_mask
        #   - Generate next token (greedy: argmax from logits[i, real_prompt_len - 1, :])
        #   - Get past_key_values for the request with self._get_past_for_request
        #   - Save state: current_len, input_ids (real part only), attention_mask, past_key_values
        #   - Set generated_tokens, num_generated, is_finished
        raise NotImplementedError("TODO: Implement prefill method")

    @torch.no_grad()
    def decode(self, requests: List[Request]) -> BatchResult:
        """
        Decode phase: generate next token for each active request using KV cache.
        
        Steps:
        1. Filter active (non-finished) requests
        2. Prepare batched KV cache with RIGHT padding
        3. Create batch from last generated tokens
        4. Build attention_mask accounting for different sequence lengths
        5. Forward pass with past_key_values and cache_position
        6. Generate next token (greedy: argmax)
        7. Update request state
        
        Note: Use RIGHT padding for KV cache. Handle finished requests separately.
        """
        # TODO: Filter active requests (if none, return empty results for all)
        # TODO: Prepare batched KV cache using _prepare_past_key_values_batch
        # TODO: Create batch from last generated tokens [batch_size, 1]
        # TODO: Build attention_mask for each active request
        # TODO: Forward pass with past_key_values
        # TODO: Get next tokens (greedy: argmax from last logit)
        # TODO: Update each request state (generated_tokens, num_generated, past_key_values, etc.)
        raise NotImplementedError("TODO: Implement decode method")

    def _get_past_for_request(
        self,
        past_key_values,
        request_idx: int,
        real_seq_len: Optional[int] = None,
    ):
        if past_key_values is None:
            return None

        new_cache = DynamicCache()
        for layer_idx in range(self.model.config.num_hidden_layers):
            key   = past_key_values.key_cache[layer_idx][request_idx:request_idx+1]
            value = past_key_values.value_cache[layer_idx][request_idx:request_idx+1]

            if real_seq_len is not None and key.shape[2] > real_seq_len:
                key   = key[:, :, :real_seq_len, :]
                value = value[:, :, :real_seq_len, :]

            new_cache.update(key, value, layer_idx)
        return new_cache

    def _prepare_past_key_values_batch(self, requests: List[Request]):
        """
        Prepare batched KV cache from requests with RIGHT padding.
        
        Combines KV cache from different requests into one batch. Since requests
        may have different sequence lengths, add RIGHT padding to max_seq_len.
        """
        if not requests:
            return None

        # TODO: Create new DynamicCache for batch
        raise NotImplementedError("TODO: Implement _prepare_past_key_values_batch method")

    def _sample(self, tokens_dist: torch.Tensor, request: Request) -> int:
        # BOUNS PART - Implement sampling logic with sampling_params
        raise NotImplementedError("I won't do bonus)")

    def get_generated_text(self, request: Request) -> str:
        if not request.generated_tokens:
            return request.prompt

        full_ids = request.input_ids[0].tolist() + request.generated_tokens
        return self.tokenizer.decode(full_ids, skip_special_tokens=True)