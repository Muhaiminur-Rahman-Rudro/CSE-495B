"""Prompting strategy implementations."""

from .base import BasePromptStrategy
from .direct import DirectPrompting
from .chain_of_thought import ChainOfThoughtPrompting
from .tree_of_thought import TreeOfThoughtPrompting
from .tree_of_graph import TreeOfGraphPrompting
from .reflexion import ReflexionPrompting

__all__ = [
    "BasePromptStrategy",
    "DirectPrompting",
    "ChainOfThoughtPrompting",
    "TreeOfThoughtPrompting",
    "TreeOfGraphPrompting",
    "ReflexionPrompting",
]


def get_prompting_strategy(name: str, **kwargs):
    """Factory function to get prompting strategy by name."""
    strategies = {
        "direct": DirectPrompting,
        "cot": ChainOfThoughtPrompting,
        "tot": TreeOfThoughtPrompting,
        "tog": TreeOfGraphPrompting,
        "reflexion": ReflexionPrompting,
    }
    
    if name not in strategies:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(strategies.keys())}")
    
    return strategies[name](**kwargs)
