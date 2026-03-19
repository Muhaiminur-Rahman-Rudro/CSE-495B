"""
Direct Prompting Strategy

Standard prompting without explicit reasoning guidance.
"""

from typing import Dict, Any, Optional
from .base import BasePromptStrategy, PromptResult


class DirectPrompting(BasePromptStrategy):
    """
    Direct prompting - asks the model to answer directly without
    explicit reasoning instructions.
    """
    
    @property
    def name(self) -> str:
        return "direct"
    
    def __init__(self, model, tokenizer, generation_config: Optional[Dict[str, Any]] = None):
        super().__init__(model, tokenizer, generation_config)
        
        self.system_prompt = (
            "You are a helpful assistant. Answer the question directly and concisely."
        )
    
    def format_prompt(self, question: str, context: Optional[str] = None) -> str:
        """Format question for direct prompting."""
        prompt_parts = []
        
        if context:
            prompt_parts.append(f"Context: {context}\n")
        
        prompt_parts.append(f"Question: {question}")
        prompt_parts.append("\nAnswer:")
        
        return "".join(prompt_parts)
    
    def generate(
        self,
        question: str,
        context: Optional[str] = None,
        **kwargs
    ) -> PromptResult:
        """Generate direct answer without reasoning steps."""
        
        prompt = self.format_prompt(question, context)
        raw_output = self._generate_text(prompt, **kwargs)
        
        # Extract answer
        final_answer = self.extract_answer(raw_output)
        
        return PromptResult(
            final_answer=final_answer,
            reasoning_trace=[],  # No explicit reasoning for direct prompting
            raw_output=raw_output,
            num_reasoning_steps=0,
            metadata={
                "strategy": self.name,
                "prompt": prompt,
            }
        )
