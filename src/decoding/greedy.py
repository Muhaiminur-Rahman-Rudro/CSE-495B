"""
Greedy Decoding Strategy

Always selects the token with the highest probability.
"""

from typing import Dict, Any
from .base import BaseDecodingStrategy, DecodingConfig


class GreedyDecoding(BaseDecodingStrategy):
    """
    Greedy decoding - always picks the most probable next token.
    
    Pros:
    - Deterministic output
    - Fast (single pass)
    - Good for factual/deterministic tasks
    
    Cons:
    - Can get stuck in repetitive loops
    - May miss better solutions available via different paths
    - Less diverse outputs
    """
    
    def __init__(
        self,
        max_new_tokens: int = 512,
        repetition_penalty: float = 1.2,
        no_repeat_ngram_size: int = 3,
    ):
        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
    
    @property
    def name(self) -> str:
        return "greedy"
    
    def get_config(self) -> DecodingConfig:
        return DecodingConfig(
            max_new_tokens=self.max_new_tokens,
            do_sample=False,  # Key: no sampling
            num_beams=1,  # No beam search
            temperature=1.0,  # Not used with do_sample=False
            repetition_penalty=self.repetition_penalty,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
        )
