"""
Reflexion Prompting Strategy

Based on: Shinn et al. (2023), "Reflexion: Language Agents with Verbal 
Reinforcement Learning"
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from .base import BasePromptStrategy, PromptResult


@dataclass
class ReflexionIteration:
    """Stores information about each reflexion iteration."""
    attempt_number: int
    response: str
    reflection: str
    is_correct: Optional[bool]


class ReflexionPrompting(BasePromptStrategy):
    """
    Reflexion prompting - iteratively refines answers through
    self-reflection and feedback.
    """
    
    @property
    def name(self) -> str:
        return "reflexion"
    
    def __init__(
        self,
        model,
        tokenizer,
        generation_config: Optional[Dict[str, Any]] = None,
        max_iterations: int = 3,
        use_external_feedback: bool = False,
    ):
        super().__init__(model, tokenizer, generation_config)
        
        self.max_iterations = max_iterations
        self.use_external_feedback = use_external_feedback
    
    def format_prompt(self, question: str, context: Optional[str] = None) -> str:
        """Format initial prompt."""
        prompt_parts = []
        
        if context:
            prompt_parts.append(f"Context: {context}\n")
        
        prompt_parts.append(f"Question: {question}")
        prompt_parts.append("\nProvide a detailed solution:")
        
        return "\n".join(prompt_parts)
    
    def _generate_initial_response(self, question: str, context: Optional[str] = None) -> str:
        """Generate initial response attempt."""
        prompt = self.format_prompt(question, context)
        return self._generate_text(prompt)
    
    def _generate_reflection(
        self,
        question: str,
        response: str,
        previous_reflections: List[str] = None
    ) -> str:
        """Generate self-reflection on the response."""
        prev_reflections_str = ""
        if previous_reflections:
            prev_reflections_str = "\n\nPrevious reflections:\n" + "\n".join(
                [f"- {r}" for r in previous_reflections]
            )
        
        prompt = f"""Analyze the following response to identify any errors, gaps in reasoning, 
or areas for improvement.

Question: {question}

Response: {response}
{prev_reflections_str}

Reflection (identify specific errors or improvements needed):"""
        
        return self._generate_text(prompt, max_new_tokens=256)
    
    def _generate_refined_response(
        self,
        question: str,
        previous_response: str,
        reflection: str,
        context: Optional[str] = None
    ) -> str:
        """Generate improved response based on reflection."""
        context_str = f"Context: {context}\n\n" if context else ""
        
        prompt = f"""{context_str}Question: {question}

Previous attempt: {previous_response}

Reflection on errors: {reflection}

Based on the reflection, provide an improved solution that addresses the identified issues:"""
        
        return self._generate_text(prompt)
    
    def _check_answer_quality(self, question: str, response: str) -> bool:
        """Self-evaluate if the answer is likely correct."""
        prompt = f"""Evaluate if the following response correctly and completely answers the question.

Question: {question}

Response: {response}

Is this response correct and complete? Answer with just 'yes' or 'no':"""
        
        output = self._generate_text(prompt, max_new_tokens=10)
        return 'yes' in output.lower()
    
    def generate(
        self,
        question: str,
        context: Optional[str] = None,
        ground_truth: Optional[str] = None,  # For external feedback
        **kwargs
    ) -> PromptResult:
        """Generate answer with iterative reflexion."""
        
        iterations: List[ReflexionIteration] = []
        reflections: List[str] = []
        
        # Initial attempt
        current_response = self._generate_initial_response(question, context)
        
        for i in range(self.max_iterations):
            # Check if answer seems correct (or use external feedback)
            if self.use_external_feedback and ground_truth:
                is_correct = self.extract_answer(current_response).strip() == ground_truth.strip()
            else:
                is_correct = self._check_answer_quality(question, current_response)
            
            # Generate reflection
            reflection = self._generate_reflection(question, current_response, reflections)
            reflections.append(reflection)
            
            iterations.append(ReflexionIteration(
                attempt_number=i + 1,
                response=current_response,
                reflection=reflection,
                is_correct=is_correct
            ))
            
            # Stop if answer is correct
            if is_correct:
                break
            
            # Generate refined response
            current_response = self._generate_refined_response(
                question, current_response, reflection, context
            )
        
        # Compile reasoning trace
        reasoning_trace = []
        for iteration in iterations:
            reasoning_trace.append(f"Attempt {iteration.attempt_number}: {iteration.response[:200]}...")
            reasoning_trace.append(f"Reflection: {iteration.reflection}")
        
        final_answer = self.extract_answer(current_response)
        
        return PromptResult(
            final_answer=final_answer,
            reasoning_trace=reasoning_trace,
            raw_output=current_response,
            num_reasoning_steps=len(iterations) * 2,  # attempts + reflections
            metadata={
                "strategy": self.name,
                "num_iterations": len(iterations),
                "iterations": [
                    {
                        "attempt": it.attempt_number,
                        "response": it.response,
                        "reflection": it.reflection,
                        "is_correct": it.is_correct
                    }
                    for it in iterations
                ],
                "final_iteration_correct": iterations[-1].is_correct if iterations else None,
            }
        )
