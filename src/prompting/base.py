"""
Base class for prompting strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class PromptResult:
    """Container for prompting results."""
    final_answer: str
    reasoning_trace: List[str]
    raw_output: str
    num_reasoning_steps: int
    metadata: Dict[str, Any]


class BasePromptStrategy(ABC):
    """
    Abstract base class for all prompting strategies.
    
    All prompting strategies should inherit from this class and implement
    the required methods.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        generation_config: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config or {}
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the prompting strategy."""
        pass
    
    @abstractmethod
    def format_prompt(self, question: str, context: Optional[str] = None) -> str:
        """
        Format the input question into a prompt for the model.
        
        Args:
            question: The question to answer
            context: Optional context for the question
            
        Returns:
            Formatted prompt string
        """
        pass
    
    @abstractmethod
    def generate(
        self,
        question: str,
        context: Optional[str] = None,
        **kwargs
    ) -> PromptResult:
        """
        Generate a response using this prompting strategy.
        
        Args:
            question: The question to answer
            context: Optional context for the question
            **kwargs: Additional generation parameters
            
        Returns:
            PromptResult containing the answer and reasoning trace
        """
        pass
    
    def _generate_text(self, prompt: str, **kwargs) -> str:
        """Helper method to generate text from the model."""
        merged_config = {**self.generation_config, **kwargs}
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=merged_config.get("max_input_length", 2048),
        ).to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=merged_config.get("max_new_tokens", 512),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **{k: v for k, v in merged_config.items() 
               if k not in ["max_input_length", "max_new_tokens"]},
        )
        
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    def extract_answer(self, text: str) -> str:
        """
        Extract the final answer from generated text.
        Override in subclasses for custom extraction logic.
        """
        # Look for common answer patterns
        import re
        
        patterns = [
            r"(?:the answer is|answer:)\s*(.+?)(?:\.|$)",
            r"(?:therefore,?|thus,?|so,?)\s*(.+?)(?:\.|$)",
            r"####\s*(.+?)(?:\n|$)",  # GSM8K format
        ]
        
        text_lower = text.lower()
        for pattern in patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Return last line as fallback
        lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
        return lines[-1] if lines else text.strip()
