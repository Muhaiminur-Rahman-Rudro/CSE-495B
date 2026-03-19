"""
Unit tests for evaluation metrics.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import (
    normalize_answer,
    extract_number,
    compute_accuracy,
    compute_reasoning_metrics,
)


class TestNormalizeAnswer:
    """Tests for answer normalization."""
    
    def test_lowercase(self):
        assert normalize_answer("HELLO") == "hello"
    
    def test_strip_punctuation(self):
        assert normalize_answer("answer.") == "answer"
        assert normalize_answer("answer!") == "answer"
    
    def test_remove_prefix(self):
        assert normalize_answer("The answer is 42") == "42"
        assert normalize_answer("Therefore 100") == "100"
    
    def test_whitespace(self):
        assert normalize_answer("  hello   world  ") == "hello world"


class TestExtractNumber:
    """Tests for number extraction."""
    
    def test_gsm8k_format(self):
        assert extract_number("The answer is #### 42") == 42
        assert extract_number("#### 100") == 100
    
    def test_with_commas(self):
        assert extract_number("The result is 1,234") == 1234
    
    def test_decimal(self):
        assert extract_number("3.14") == 3.14
    
    def test_negative(self):
        assert extract_number("-5") == -5
    
    def test_last_number(self):
        # Should return last number
        assert extract_number("First 10, then 20, finally 30") == 30


class TestComputeAccuracy:
    """Tests for accuracy computation."""
    
    def test_perfect_accuracy(self):
        predictions = ["42", "100", "yes"]
        references = ["42", "100", "yes"]
        
        result = compute_accuracy(predictions, references)
        assert result["accuracy"] == 1.0
        assert result["exact_match"] == 1.0
    
    def test_zero_accuracy(self):
        predictions = ["wrong", "wrong", "wrong"]
        references = ["right", "correct", "yes"]
        
        result = compute_accuracy(predictions, references)
        assert result["accuracy"] == 0.0
    
    def test_partial_accuracy(self):
        predictions = ["42", "wrong"]
        references = ["42", "100"]
        
        result = compute_accuracy(predictions, references)
        assert result["accuracy"] == 0.5
    
    def test_arithmetic_task(self):
        predictions = ["The answer is 42", "#### 100"]
        references = ["42", "100"]
        
        result = compute_accuracy(predictions, references, task_type="arithmetic")
        assert result["accuracy"] == 1.0


class TestComputeReasoningMetrics:
    """Tests for reasoning metrics computation."""
    
    def test_average_steps(self):
        traces = [
            ["Step 1", "Step 2"],
            ["Step 1", "Step 2", "Step 3", "Step 4"],
        ]
        outputs = ["Output 1", "Output 2"]
        
        result = compute_reasoning_metrics(traces, outputs)
        assert result["avg_reasoning_steps"] == 3.0
    
    def test_empty_traces(self):
        traces = [[], []]
        outputs = ["Output 1", "Output 2"]
        
        result = compute_reasoning_metrics(traces, outputs)
        assert result["avg_reasoning_steps"] == 0.0
        assert result["pct_with_reasoning"] == 0.0
    
    def test_mixed_traces(self):
        traces = [[], ["Step 1", "Step 2"]]
        outputs = ["Short", "Longer output here"]
        
        result = compute_reasoning_metrics(traces, outputs)
        assert result["pct_with_reasoning"] == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
