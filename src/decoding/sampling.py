"""
Sampling-based Decoding Strategies

Top-k, Top-p (nucleus), and combined sampling approaches.
"""

from typing import Dict, Any
from .base import BaseDecodingStrategy, DecodingConfig


class TopKSampling(BaseDecodingStrategy):
    """
    Top-k sampling - samples from the k most probable tokens.
    
    Pros:
    - More diverse than greedy/beam search
    - Prevents sampling very low probability tokens
    - Configurable diversity via k
    
    Cons:
    - Fixed k may be too restrictive or too permissive
    - Doesn't adapt to probability distribution shape
    """
    
    def __init__(
        self,
        top_k: int = 50,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
        repetition_penalty: float = 1.1,
    ):
        self.top_k = top_k
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty
    
    @property
    def name(self) -> str:
        return f"top_k_{self.top_k}_t{self.temperature}"
    
    def get_config(self) -> DecodingConfig:
        return DecodingConfig(
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            top_k=self.top_k,
            top_p=1.0,  # Disable top-p
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
        )


class TopPSampling(BaseDecodingStrategy):
    """
    Top-p (nucleus) sampling - samples from smallest set of tokens 
    whose cumulative probability exceeds p.
    
    Pros:
    - Adapts to probability distribution
    - More natural text generation
    - Better balance of quality and diversity
    
    Cons:
    - Can occasionally include unlikely tokens
    - Less predictable output length
    """
    
    def __init__(
        self,
        top_p: float = 0.9,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
        repetition_penalty: float = 1.1,
    ):
        self.top_p = top_p
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty
    
    @property
    def name(self) -> str:
        return f"top_p_{self.top_p}_t{self.temperature}"
    
    def get_config(self) -> DecodingConfig:
        return DecodingConfig(
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            top_k=0,  # Disable top-k
            top_p=self.top_p,
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
        )


class CombinedSampling(BaseDecodingStrategy):
    """
    Combined top-k and top-p sampling - applies both filters.
    
    This is often the best approach, combining the benefits of both:
    - Top-k provides hard cutoff on number of candidates
    - Top-p adapts to probability distribution
    """
    
    def __init__(
        self,
        top_k: int = 50,
        top_p: float = 0.9,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
        repetition_penalty: float = 1.1,
    ):
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty
    
    @property
    def name(self) -> str:
        return f"combined_k{self.top_k}_p{self.top_p}_t{self.temperature}"
    
    def get_config(self) -> DecodingConfig:
        return DecodingConfig(
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            top_k=self.top_k,
            top_p=self.top_p,
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
        )


class TemperatureSampling(BaseDecodingStrategy):
    """
    Pure temperature-based sampling without top-k/top-p filtering.
    
    Temperature controls the sharpness of the probability distribution:
    - T < 1: Sharper distribution (more confident)
    - T = 1: Original distribution
    - T > 1: Flatter distribution (more random)
    """
    
    def __init__(
        self,
        temperature: float = 1.0,
        max_new_tokens: int = 512,
        repetition_penalty: float = 1.1,
    ):
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty
    
    @property
    def name(self) -> str:
        return f"temperature_{self.temperature}"
    
    def get_config(self) -> DecodingConfig:
        return DecodingConfig(
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            top_k=0,
            top_p=1.0,
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
        )
