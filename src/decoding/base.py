"""
Base class for decoding strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class DecodingConfig:
    """Configuration for decoding strategies."""
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    num_beams: int = 1
    do_sample: bool = True
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    early_stopping: bool = False
    num_return_sequences: int = 1
    
    # Additional parameters
    no_repeat_ngram_size: int = 0
    min_new_tokens: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for HuggingFace generate()."""
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "num_beams": self.num_beams,
            "do_sample": self.do_sample,
            "repetition_penalty": self.repetition_penalty,
            "length_penalty": self.length_penalty,
            "early_stopping": self.early_stopping,
            "num_return_sequences": self.num_return_sequences,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "min_new_tokens": self.min_new_tokens,
        }


class BaseDecodingStrategy(ABC):
    """
    Abstract base class for decoding strategies.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the decoding strategy."""
        pass
    
    @abstractmethod
    def get_config(self) -> DecodingConfig:
        """Return the decoding configuration."""
        pass
    
    def get_generation_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for HuggingFace model.generate()."""
        return self.get_config().to_dict()
    
    def __repr__(self) -> str:
        config = self.get_config()
        return f"{self.__class__.__name__}(config={config})"
