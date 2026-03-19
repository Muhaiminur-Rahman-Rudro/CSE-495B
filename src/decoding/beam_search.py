"""
Beam Search Decoding Strategy

Maintains multiple hypotheses and selects the best overall sequence.
"""

from typing import Dict, Any
from .base import BaseDecodingStrategy, DecodingConfig


class BeamSearchDecoding(BaseDecodingStrategy):
    """
    Beam search decoding - maintains top-k hypotheses at each step.
    
    Pros:
    - Often finds higher probability sequences than greedy
    - Can explore multiple paths simultaneously
    - Good for translation and structured generation
    
    Cons:
    - Slower than greedy (beam_width times more computation)
    - Can still be repetitive
    - May prefer shorter sequences
    """
    
    def __init__(
        self,
        num_beams: int = 5,
        max_new_tokens: int = 512,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
        no_repeat_ngram_size: int = 3,
        num_return_sequences: int = 1,
    ):
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.num_return_sequences = min(num_return_sequences, num_beams)
    
    @property
    def name(self) -> str:
        return f"beam_search_b{self.num_beams}"
    
    def get_config(self) -> DecodingConfig:
        return DecodingConfig(
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            num_beams=self.num_beams,
            length_penalty=self.length_penalty,
            early_stopping=self.early_stopping,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            num_return_sequences=self.num_return_sequences,
        )


class DiverseBeamSearchDecoding(BaseDecodingStrategy):
    """
    Diverse beam search - encourages diversity between beam groups.
    """
    
    def __init__(
        self,
        num_beams: int = 6,
        num_beam_groups: int = 3,
        diversity_penalty: float = 0.5,
        max_new_tokens: int = 512,
        length_penalty: float = 1.0,
    ):
        self.num_beams = num_beams
        self.num_beam_groups = num_beam_groups
        self.diversity_penalty = diversity_penalty
        self.max_new_tokens = max_new_tokens
        self.length_penalty = length_penalty
    
    @property
    def name(self) -> str:
        return f"diverse_beam_search_b{self.num_beams}_g{self.num_beam_groups}"
    
    def get_config(self) -> DecodingConfig:
        config = DecodingConfig(
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            num_beams=self.num_beams,
            length_penalty=self.length_penalty,
        )
        return config
    
    def get_generation_kwargs(self) -> Dict[str, Any]:
        kwargs = super().get_generation_kwargs()
        kwargs["num_beam_groups"] = self.num_beam_groups
        kwargs["diversity_penalty"] = self.diversity_penalty
        return kwargs
