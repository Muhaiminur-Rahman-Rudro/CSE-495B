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
        ...
    
    def _generate_text(self, prompt: str, **kwargs) -> str:
        """Helper method to generate text from the model."""
        merged_config = {**self.generation_config, **kwargs}

        max_input = merged_config.pop("max_input_length", 2048)
        max_new = merged_config.pop("max_new_tokens", 512)

        # Use chat template when the tokenizer supports it (e.g. Qwen-Instruct).
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            input_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_input,
            ).to(self.model.device)
        else:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_input,
            ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **{k: v for k, v in merged_config.items()
               if k not in ["max_input_length", "max_new_tokens"]},
        )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Strip repeated blocks that 4-bit quantised models sometimes produce.
        text = self._strip_repetitions(text)
        return text

    # ------------------------------------------------------------------
    @staticmethod
    def _strip_repetitions(text: str, min_block: int = 60) -> str:
        """Remove large repeated blocks from generated text."""
        if len(text) < min_block * 2:
            return text
        # Try to find a block that repeats immediately after itself.
        for size in range(len(text) // 2, min_block - 1, -1):
            block = text[:size]
            if text[size: size + len(block)] == block:
                return block
        return text

    # ------------------------------------------------------------------
    def extract_answer(self, text: str) -> str:
        """
        Extract the final answer from generated text.
        Override in subclasses for custom extraction logic.
        """
        import re

        # 1) GSM8K #### format
        m = re.search(r'####\s*([^\n]+)', text)
        if m:
            return m.group(1).strip().rstrip('.,;:!?')

        # 2) "the answer is <X>" / "answer: <X>"
        m = re.search(
            r'(?:the answer is|answer:)\s*(.+?)(?:\.|,|\n|$)',
            text, re.IGNORECASE,
        )
        if m:
            return m.group(1).strip().rstrip('.,;:!?')

        # 3) "therefore / thus / so" sentence
        m = re.search(
            r'(?:therefore|thus|so),?\s*(?:the answer is\s*)?(.+?)(?:\.|,|\n|$)',
            text, re.IGNORECASE,
        )
        if m:
            return m.group(1).strip().rstrip('.,;:!?')

        # 4) Last number in the text (arithmetic fallback)
        numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', text)
        if numbers:
            return numbers[-1].replace(',', '')

        # 5) Last non-empty line
        lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
        return lines[-1] if lines else text.strip()
