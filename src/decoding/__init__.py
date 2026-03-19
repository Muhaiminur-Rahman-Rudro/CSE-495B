"""Decoding strategy implementations."""

from .base import BaseDecodingStrategy, DecodingConfig
from .greedy import GreedyDecoding
from .beam_search import BeamSearchDecoding
from .sampling import TopKSampling, TopPSampling, CombinedSampling

__all__ = [
    "BaseDecodingStrategy",
    "DecodingConfig",
    "GreedyDecoding",
    "BeamSearchDecoding",
    "TopKSampling",
    "TopPSampling",
    "CombinedSampling",
]


def get_decoding_strategy(name: str, **kwargs) -> BaseDecodingStrategy:
    """Factory function to get decoding strategy by name."""
    strategies = {
        "greedy": GreedyDecoding,
        "beam_search": BeamSearchDecoding,
        "top_k": TopKSampling,
        "top_p": TopPSampling,
        "combined": CombinedSampling,
    }
    
    if name not in strategies:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(strategies.keys())}")
    
    return strategies[name](**kwargs)
