"""
Chain-of-Thought (CoT) Prompting Strategy

Based on: Wei et al. (2022), "Chain of Thought Prompting Elicits Reasoning 
in Large Language Models"
"""

from typing import Dict, Any, Optional, List
from .base import BasePromptStrategy, PromptResult
import re


class ChainOfThoughtPrompting(BasePromptStrategy):
    """
    Chain-of-Thought prompting - encourages step-by-step reasoning
    before arriving at the final answer.
    """
    
    @property
    def name(self) -> str:
        return "chain_of_thought"
    
    def __init__(
        self,
        model,
        tokenizer,
        generation_config: Optional[Dict[str, Any]] = None,
        use_few_shot: bool = False,
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
    ):
        super().__init__(model, tokenizer, generation_config)
        
        self.use_few_shot = use_few_shot
        self.few_shot_examples = few_shot_examples or self._default_few_shot_examples()
        
        self.cot_trigger = "Let's think step by step."
        self.answer_trigger = "Therefore, the answer is"
    
    def _default_few_shot_examples(self) -> List[Dict[str, str]]:
        """Default few-shot examples for arithmetic reasoning."""
        return [
            {
                "question": "Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?",
                "reasoning": "Roger started with 5 balls. 2 cans of 3 tennis balls each is 2 * 3 = 6 tennis balls. 5 + 6 = 11.",
                "answer": "11"
            },
            {
                "question": "The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?",
                "reasoning": "The cafeteria had 23 apples originally. They used 20 to make lunch. So they had 23 - 20 = 3. They bought 6 more apples, so they have 3 + 6 = 9.",
                "answer": "9"
            },
        ]
    
    def format_prompt(self, question: str, context: Optional[str] = None) -> str:
        """Format question with CoT trigger."""
        prompt_parts = []
        
        # Add few-shot examples if enabled
        if self.use_few_shot:
            for example in self.few_shot_examples:
                prompt_parts.append(f"Q: {example['question']}")
                prompt_parts.append(f"A: {self.cot_trigger} {example['reasoning']} {self.answer_trigger} {example['answer']}.\n")
        
        # Add context if provided
        if context:
            prompt_parts.append(f"Context: {context}\n")
        
        # Add the actual question with CoT trigger
        prompt_parts.append(f"Q: {question}")
        prompt_parts.append(f"A: {self.cot_trigger}")
        
        return "\n".join(prompt_parts)
    
    def _parse_reasoning_steps(self, text: str) -> List[str]:
        """Parse reasoning steps from generated text."""
        # Split by common step indicators
        step_patterns = [
            r'(?:Step \d+[:.]\s*)',
            r'(?:First,?\s*|Second,?\s*|Third,?\s*|Then,?\s*|Next,?\s*|Finally,?\s*)',
            r'(?:\d+\)\s*|\d+\.\s*)',
        ]
        
        # Try to split by sentences if no clear step indicators
        steps = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and not sentence.lower().startswith(('therefore', 'thus', 'so the answer')):
                steps.append(sentence)
        
        return steps
    
    def generate(
        self,
        question: str,
        context: Optional[str] = None,
        **kwargs
    ) -> PromptResult:
        """Generate answer with chain-of-thought reasoning."""
        
        prompt = self.format_prompt(question, context)
        raw_output = self._generate_text(prompt, **kwargs)
        
        # Parse reasoning and answer
        reasoning_steps = self._parse_reasoning_steps(raw_output)
        final_answer = self.extract_answer(raw_output)
        
        return PromptResult(
            final_answer=final_answer,
            reasoning_trace=reasoning_steps,
            raw_output=raw_output,
            num_reasoning_steps=len(reasoning_steps),
            metadata={
                "strategy": self.name,
                "prompt": prompt,
                "use_few_shot": self.use_few_shot,
            }
        )
