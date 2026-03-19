"""
Evaluation Metrics for Reasoning Tasks

Includes accuracy, reasoning trace analysis, and hallucination detection.
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
import re
import numpy as np


def _infer_reasoning_steps_from_output(text: str) -> int:
    """Infer reasoning step count from raw text when explicit trace is missing."""
    if not text:
        return 0

    cleaned = text.strip()
    if not cleaned:
        return 0

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]

    marker_pattern = re.compile(
        r"^(step\s*\d+\b|\d+[\).]\s+|first\b|second\b|third\b|then\b|next\b|finally\b|because\b)",
        re.IGNORECASE,
    )
    marker_count = sum(1 for line in lines if marker_pattern.match(line))
    if marker_count > 0:
        return marker_count

    sentence_candidates = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", cleaned)
        if s.strip()
    ]
    if len(sentence_candidates) >= 2:
        return min(len(sentence_candidates), 12)

    # Single line with operators often indicates compressed reasoning.
    if re.search(r"[=+\-*/]", cleaned):
        return 1

    return 0


def get_reasoning_step_count(reasoning_trace: List[str], raw_output: str) -> int:
    """Use explicit trace when available, otherwise infer from generated text."""
    trace_len = len(reasoning_trace or [])
    if trace_len > 0:
        return trace_len
    return _infer_reasoning_steps_from_output(raw_output)


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    accuracy: float
    exact_match: float
    reasoning_metrics: Dict[str, float]
    hallucination_metrics: Dict[str, float]
    per_sample_results: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def summary(self) -> str:
        """Return a summary string of results."""
        lines = [
            "=" * 50,
            "Evaluation Results",
            "=" * 50,
            f"Accuracy: {self.accuracy:.2%}",
            f"Exact Match: {self.exact_match:.2%}",
            "",
            "Reasoning Metrics:",
        ]
        for k, v in self.reasoning_metrics.items():
            lines.append(f"  {k}: {v:.3f}")
        
        lines.append("")
        lines.append("Hallucination Metrics:")
        for k, v in self.hallucination_metrics.items():
            lines.append(f"  {k}: {v:.3f}")
        
        lines.append("=" * 50)
        return "\n".join(lines)


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    # Remove extra whitespace
    answer = " ".join(answer.split())
    
    # Convert to lowercase
    answer = answer.lower()
    
    # Remove punctuation at the end
    answer = answer.rstrip('.,!?;:')
    
    # Remove common prefixes
    prefixes = ['the answer is', 'answer:', 'therefore', 'thus', 'so']
    for prefix in prefixes:
        if answer.startswith(prefix):
            answer = answer[len(prefix):].strip()
    
    return answer.strip()


def extract_number(text: str) -> Optional[float]:
    """Extract numeric answer from text."""
    # Handle GSM8K format: #### answer
    match = re.search(r'####\s*([\d,.-]+)', text)
    if match:
        return float(match.group(1).replace(',', ''))
    
    # Try to find any number
    numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', text)
    if numbers:
        # Return the last number (usually the final answer)
        return float(numbers[-1].replace(',', ''))
    
    return None


def compute_accuracy(
    predictions: List[str],
    references: List[str],
    task_type: str = "general"
) -> Dict[str, float]:
    """
    Compute accuracy metrics.
    
    Args:
        predictions: List of predicted answers
        references: List of ground truth answers
        task_type: Type of task (arithmetic, logic, reading)
        
    Returns:
        Dictionary with accuracy metrics
    """
    assert len(predictions) == len(references), "Predictions and references must have same length"
    
    exact_matches = 0
    correct = 0
    
    for pred, ref in zip(predictions, references):
        pred_norm = normalize_answer(pred)
        ref_norm = normalize_answer(ref)
        
        # Exact match
        if pred_norm == ref_norm:
            exact_matches += 1
            correct += 1
            continue
        
        # For arithmetic tasks, compare numbers
        if task_type == "arithmetic":
            pred_num = extract_number(pred)
            ref_num = extract_number(ref)
            
            if pred_num is not None and ref_num is not None:
                if abs(pred_num - ref_num) < 1e-6:
                    correct += 1
                    continue
        
        # Fuzzy match for non-empty strings only.
        if pred_norm and ref_norm and (ref_norm in pred_norm or pred_norm in ref_norm):
            correct += 1
    
    n = len(predictions)
    return {
        "accuracy": correct / n if n > 0 else 0,
        "exact_match": exact_matches / n if n > 0 else 0,
        "num_samples": n,
    }


def compute_reasoning_metrics(
    reasoning_traces: List[List[str]],
    raw_outputs: List[str]
) -> Dict[str, float]:
    """
    Compute metrics about reasoning quality.
    
    Args:
        reasoning_traces: List of reasoning step lists
        raw_outputs: Raw model outputs
        
    Returns:
        Dictionary with reasoning metrics
    """
    num_steps = [
        get_reasoning_step_count(trace, output)
        for trace, output in zip(reasoning_traces, raw_outputs)
    ]
    output_lengths = [len(output.split()) for output in raw_outputs]
    
    metrics = {
        "avg_reasoning_steps": np.mean(num_steps) if num_steps else 0,
        "std_reasoning_steps": np.std(num_steps) if num_steps else 0,
        "min_reasoning_steps": min(num_steps) if num_steps else 0,
        "max_reasoning_steps": max(num_steps) if num_steps else 0,
        "avg_output_length": np.mean(output_lengths) if output_lengths else 0,
        "pct_with_reasoning": sum(1 for n in num_steps if n > 0) / len(num_steps) if num_steps else 0,
    }
    
    return metrics


def compute_hallucination_score(
    outputs: List[str],
    contexts: Optional[List[str]] = None,
    questions: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute hallucination-related metrics.
    
    This is a simplified heuristic-based approach. For production use,
    consider using dedicated hallucination detection models.
    
    Detects:
    - Self-contradictions in reasoning
    - Unsupported claims (when context is provided)
    - Confidence indicators without justification
    """
    metrics = {
        "contradiction_rate": 0.0,
        "unsupported_claim_rate": 0.0,
        "overconfidence_rate": 0.0,
    }
    
    contradiction_patterns = [
        (r'however.*but', 'contradiction'),
        (r'is (\d+).*is (\d+)', 'number_inconsistency'),
        (r'therefore.*however', 'logical_contradiction'),
    ]
    
    overconfidence_markers = [
        'definitely', 'certainly', 'obviously', 'clearly',
        'without a doubt', 'absolutely', '100%'
    ]
    
    contradictions = 0
    overconfident = 0
    
    for output in outputs:
        output_lower = output.lower()
        
        # Check for contradictions
        for pattern, _ in contradiction_patterns:
            if re.search(pattern, output_lower):
                contradictions += 1
                break
        
        # Check for overconfidence
        for marker in overconfidence_markers:
            if marker in output_lower:
                overconfident += 1
                break
    
    n = len(outputs)
    if n > 0:
        metrics["contradiction_rate"] = contradictions / n
        metrics["overconfidence_rate"] = overconfident / n
    
    return metrics


def compute_consistency(
    outputs: List[List[str]],
) -> Dict[str, float]:
    """
    Compute consistency across multiple runs of the same input.
    
    Args:
        outputs: List of output lists (multiple runs per input)
        
    Returns:
        Consistency metrics
    """
    if not outputs or not outputs[0]:
        return {"consistency_rate": 0.0}
    
    consistent = 0
    total = len(outputs)
    
    for runs in outputs:
        if len(runs) <= 1:
            continue
        
        # Check if all runs produced the same answer
        normalized = [normalize_answer(r) for r in runs]
        if len(set(normalized)) == 1:
            consistent += 1
    
    return {
        "consistency_rate": consistent / total if total > 0 else 0,
    }
