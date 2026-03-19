"""
Tree-of-Thought (ToT) Prompting Strategy

Based on: Yao et al. (2023), "Tree of Thoughts: Deliberate Problem Solving 
with Large Language Models"
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from .base import BasePromptStrategy, PromptResult
import re


@dataclass
class ThoughtNode:
    """Represents a node in the thought tree."""
    thought: str
    evaluation_score: float
    children: List['ThoughtNode']
    depth: int
    path: List[str]


class TreeOfThoughtPrompting(BasePromptStrategy):
    """
    Tree-of-Thought prompting - explores multiple reasoning paths
    and uses self-evaluation to select the best path.
    """
    
    @property
    def name(self) -> str:
        return "tree_of_thought"
    
    def __init__(
        self,
        model,
        tokenizer,
        generation_config: Optional[Dict[str, Any]] = None,
        num_branches: int = 3,
        max_depth: int = 3,
        evaluation_strategy: str = "vote",  # "vote" or "score"
    ):
        super().__init__(model, tokenizer, generation_config)
        
        self.num_branches = num_branches
        self.max_depth = max_depth
        self.evaluation_strategy = evaluation_strategy
    
    def format_prompt(self, question: str, context: Optional[str] = None) -> str:
        """Format base prompt for ToT."""
        prompt_parts = []
        
        if context:
            prompt_parts.append(f"Context: {context}\n")
        
        prompt_parts.append(f"Problem: {question}")
        
        return "\n".join(prompt_parts)
    
    def _generate_thoughts(
        self,
        question: str,
        current_state: str,
        num_thoughts: int = 3
    ) -> List[str]:
        """Generate multiple possible next thoughts."""
        prompt = f"""Given the problem and current progress, propose {num_thoughts} distinct possible next steps.

Problem: {question}

Current progress:
{current_state if current_state else "Starting fresh."}

List {num_thoughts} different approaches or next steps to consider. Be specific and distinct.

Possible next steps:
1."""
        
        output = self._generate_text(prompt, temperature=0.7, top_p=0.9)
        
        # Parse numbered thoughts
        thoughts = []
        lines = output.split('\n')
        current_thought = []
        
        for line in lines:
            if re.match(r'^\d+[.)]\s*', line):
                if current_thought:
                    thoughts.append(' '.join(current_thought).strip())
                current_thought = [re.sub(r'^\d+[.)]\s*', '', line)]
            elif current_thought:
                current_thought.append(line)
        
        if current_thought:
            thoughts.append(' '.join(current_thought).strip())
        
        return thoughts[:num_thoughts]
    
    def _evaluate_thought(
        self,
        question: str,
        thought_path: List[str],
    ) -> float:
        """Evaluate a thought path using self-evaluation."""
        path_str = "\n".join([f"Step {i+1}: {t}" for i, t in enumerate(thought_path)])
        
        if self.evaluation_strategy == "score":
            prompt = f"""Evaluate the following reasoning path for solving the problem.

Problem: {question}

Reasoning path:
{path_str}

Rate this reasoning path from 1-10 based on:
- Correctness of logic
- Progress toward the solution
- Clarity of reasoning

Score (1-10):"""
            
            output = self._generate_text(prompt, max_new_tokens=10)
            
            # Extract score
            match = re.search(r'(\d+(?:\.\d+)?)', output)
            if match:
                return min(10, max(1, float(match.group(1)))) / 10
            return 0.5
        
        else:  # vote strategy
            prompt = f"""Does this reasoning path make progress toward solving the problem?

Problem: {question}

Reasoning path:
{path_str}

Answer (yes/no):"""
            
            output = self._generate_text(prompt, max_new_tokens=10)
            return 1.0 if 'yes' in output.lower() else 0.0
    
    def _search_tree(
        self,
        question: str,
        context: Optional[str],
        max_iterations: int = 10
    ) -> Tuple[List[str], List[ThoughtNode]]:
        """Perform tree search to find best reasoning path."""
        all_nodes = []
        best_path = []
        best_score = 0
        
        # Initialize with root thoughts
        initial_thoughts = self._generate_thoughts(question, "", self.num_branches)
        
        # BFS-style search with pruning
        frontier = []
        for thought in initial_thoughts:
            score = self._evaluate_thought(question, [thought])
            node = ThoughtNode(
                thought=thought,
                evaluation_score=score,
                children=[],
                depth=1,
                path=[thought]
            )
            frontier.append(node)
            all_nodes.append(node)
        
        iterations = 0
        while frontier and iterations < max_iterations:
            iterations += 1
            
            # Sort by score and take top candidates
            frontier.sort(key=lambda x: x.evaluation_score, reverse=True)
            current = frontier.pop(0)
            
            # Check if this is the best path so far
            if current.evaluation_score > best_score:
                best_score = current.evaluation_score
                best_path = current.path
            
            # Stop if max depth reached
            if current.depth >= self.max_depth:
                continue
            
            # Generate children
            state = "\n".join([f"Step {i+1}: {t}" for i, t in enumerate(current.path)])
            child_thoughts = self._generate_thoughts(question, state, self.num_branches)
            
            for thought in child_thoughts:
                new_path = current.path + [thought]
                score = self._evaluate_thought(question, new_path)
                
                child_node = ThoughtNode(
                    thought=thought,
                    evaluation_score=score,
                    children=[],
                    depth=current.depth + 1,
                    path=new_path
                )
                current.children.append(child_node)
                frontier.append(child_node)
                all_nodes.append(child_node)
        
        return best_path, all_nodes
    
    def _generate_final_answer(self, question: str, reasoning_path: List[str]) -> str:
        """Generate final answer from the best reasoning path."""
        path_str = "\n".join([f"Step {i+1}: {t}" for i, t in enumerate(reasoning_path)])
        
        prompt = f"""Based on the following reasoning, provide the final answer.

Problem: {question}

Reasoning:
{path_str}

Final Answer:"""
        
        return self._generate_text(prompt, max_new_tokens=100)
    
    def generate(
        self,
        question: str,
        context: Optional[str] = None,
        **kwargs
    ) -> PromptResult:
        """Generate answer using tree-of-thought reasoning."""
        
        # Search for best reasoning path
        best_path, all_nodes = self._search_tree(question, context)
        
        # Generate final answer from best path
        if best_path:
            final_output = self._generate_final_answer(question, best_path)
            final_answer = self.extract_answer(final_output)
        else:
            final_answer = "Unable to find a valid reasoning path."
            final_output = final_answer
        
        return PromptResult(
            final_answer=final_answer,
            reasoning_trace=best_path,
            raw_output=final_output,
            num_reasoning_steps=len(best_path),
            metadata={
                "strategy": self.name,
                "num_branches": self.num_branches,
                "max_depth": self.max_depth,
                "total_nodes_explored": len(all_nodes),
                "evaluation_strategy": self.evaluation_strategy,
            }
        )
