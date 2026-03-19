"""
Reasoning Evaluator

Main evaluation class that orchestrates all metrics computation.
"""

from typing import Dict, Any, List, Optional
import json
import time
from pathlib import Path
from dataclasses import asdict
import logging

from .metrics import (
    compute_accuracy,
    compute_reasoning_metrics,
    compute_hallucination_score,
    get_reasoning_step_count,
    EvaluationResult,
)

logger = logging.getLogger(__name__)


class ReasoningEvaluator:
    """
    Comprehensive evaluator for reasoning tasks.
    
    Computes accuracy, reasoning quality, and hallucination metrics.
    """
    
    def __init__(
        self,
        task_type: str = "general",
        output_dir: Optional[str] = None,
    ):
        self.task_type = task_type
        self.output_dir = Path(output_dir) if output_dir else None
        
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate(
        self,
        predictions: List[Dict[str, Any]],
        references: List[Dict[str, Any]],
        experiment_name: str = "experiment",
    ) -> EvaluationResult:
        """
        Run full evaluation.
        
        Args:
            predictions: List of prediction dicts with keys:
                - final_answer: str
                - reasoning_trace: List[str]
                - raw_output: str
                - metadata: Dict
            references: List of reference dicts with keys:
                - answer: str
                - question: str
                - context: Optional[str]
            experiment_name: Name for this evaluation run
            
        Returns:
            EvaluationResult with all metrics
        """
        logger.info(f"Starting evaluation: {experiment_name}")
        start_time = time.time()
        
        # Extract fields
        pred_answers = [p["final_answer"] for p in predictions]
        ref_answers = [r["answer"] for r in references]
        reasoning_traces = [p.get("reasoning_trace", []) for p in predictions]
        raw_outputs = [p.get("raw_output", p["final_answer"]) for p in predictions]
        contexts = [r.get("context") for r in references]
        questions = [r.get("question") for r in references]
        
        # Compute metrics
        accuracy_metrics = compute_accuracy(
            pred_answers, ref_answers, self.task_type
        )
        
        reasoning_metrics = compute_reasoning_metrics(
            reasoning_traces, raw_outputs
        )
        
        hallucination_metrics = compute_hallucination_score(
            raw_outputs, contexts, questions
        )
        
        # Per-sample results
        per_sample_results = []
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            sample_result = {
                "idx": i,
                "question": questions[i] if questions else None,
                "predicted": pred["final_answer"],
                "reference": ref["answer"],
                "correct": pred_answers[i].strip().lower() == ref_answers[i].strip().lower(),
                "num_reasoning_steps": get_reasoning_step_count(
                    pred.get("reasoning_trace", []),
                    pred.get("raw_output", pred.get("final_answer", "")),
                ),
            }
            per_sample_results.append(sample_result)
        
        # Build result
        result = EvaluationResult(
            accuracy=accuracy_metrics["accuracy"],
            exact_match=accuracy_metrics["exact_match"],
            reasoning_metrics=reasoning_metrics,
            hallucination_metrics=hallucination_metrics,
            per_sample_results=per_sample_results,
            metadata={
                "experiment_name": experiment_name,
                "task_type": self.task_type,
                "num_samples": len(predictions),
                "evaluation_time": time.time() - start_time,
            }
        )
        
        # Save results
        if self.output_dir:
            self._save_results(result, experiment_name)
        
        logger.info(f"Evaluation complete in {result.metadata['evaluation_time']:.2f}s")
        logger.info(f"Accuracy: {result.accuracy:.2%}")
        
        return result
    
    def _save_results(self, result: EvaluationResult, experiment_name: str):
        """Save evaluation results to disk."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{experiment_name}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        # Convert to serializable format
        result_dict = {
            "accuracy": result.accuracy,
            "exact_match": result.exact_match,
            "reasoning_metrics": result.reasoning_metrics,
            "hallucination_metrics": result.hallucination_metrics,
            "per_sample_results": result.per_sample_results,
            "metadata": result.metadata,
        }
        
        with open(filepath, "w") as f:
            json.dump(result_dict, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
    
    def compare_experiments(
        self,
        results: Dict[str, EvaluationResult]
    ) -> Dict[str, Any]:
        """
        Compare results across multiple experiments.
        
        Args:
            results: Dictionary mapping experiment names to results
            
        Returns:
            Comparison summary
        """
        comparison = {
            "experiments": list(results.keys()),
            "accuracy": {name: r.accuracy for name, r in results.items()},
            "exact_match": {name: r.exact_match for name, r in results.items()},
            "avg_reasoning_steps": {
                name: r.reasoning_metrics.get("avg_reasoning_steps", 0)
                for name, r in results.items()
            },
            "hallucination_rate": {
                name: r.hallucination_metrics.get("contradiction_rate", 0)
                for name, r in results.items()
            },
        }
        
        # Find best performing
        best_accuracy = max(comparison["accuracy"].items(), key=lambda x: x[1])
        comparison["best_accuracy"] = {
            "experiment": best_accuracy[0],
            "value": best_accuracy[1]
        }
        
        return comparison
