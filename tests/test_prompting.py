"""
Unit tests for prompting strategies.
"""

import pytest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prompting.base import PromptResult
from src.prompting.direct import DirectPrompting
from src.prompting.chain_of_thought import ChainOfThoughtPrompting


class TestDirectPrompting:
    """Tests for Direct prompting strategy."""
    
    @pytest.fixture
    def mock_model(self):
        model = MagicMock()
        model.device = "cpu"
        return model
    
    @pytest.fixture
    def mock_tokenizer(self):
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 1
        tokenizer.decode.return_value = "The answer is 42."
        return tokenizer
    
    def test_name(self, mock_model, mock_tokenizer):
        strategy = DirectPrompting(mock_model, mock_tokenizer)
        assert strategy.name == "direct"
    
    def test_format_prompt(self, mock_model, mock_tokenizer):
        strategy = DirectPrompting(mock_model, mock_tokenizer)
        prompt = strategy.format_prompt("What is 2+2?")
        
        assert "Question: What is 2+2?" in prompt
        assert "Answer:" in prompt
    
    def test_format_prompt_with_context(self, mock_model, mock_tokenizer):
        strategy = DirectPrompting(mock_model, mock_tokenizer)
        prompt = strategy.format_prompt("What is the capital?", context="France is a country.")
        
        assert "Context: France is a country." in prompt
        assert "Question: What is the capital?" in prompt


class TestChainOfThoughtPrompting:
    """Tests for CoT prompting strategy."""
    
    @pytest.fixture
    def mock_model(self):
        model = MagicMock()
        model.device = "cpu"
        return model
    
    @pytest.fixture
    def mock_tokenizer(self):
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 1
        return tokenizer
    
    def test_name(self, mock_model, mock_tokenizer):
        strategy = ChainOfThoughtPrompting(mock_model, mock_tokenizer)
        assert strategy.name == "chain_of_thought"
    
    def test_format_prompt_contains_cot_trigger(self, mock_model, mock_tokenizer):
        strategy = ChainOfThoughtPrompting(mock_model, mock_tokenizer)
        prompt = strategy.format_prompt("What is 2+2?")
        
        assert "Let's think step by step" in prompt
    
    def test_format_prompt_with_few_shot(self, mock_model, mock_tokenizer):
        strategy = ChainOfThoughtPrompting(
            mock_model, mock_tokenizer, use_few_shot=True
        )
        prompt = strategy.format_prompt("What is 5+3?")
        
        # Should contain few-shot examples
        assert "Roger has 5 tennis balls" in prompt or "cafeteria" in prompt


class TestPromptResult:
    """Tests for PromptResult dataclass."""
    
    def test_creation(self):
        result = PromptResult(
            final_answer="42",
            reasoning_trace=["Step 1", "Step 2"],
            raw_output="Let me think... The answer is 42.",
            num_reasoning_steps=2,
            metadata={"strategy": "test"}
        )
        
        assert result.final_answer == "42"
        assert len(result.reasoning_trace) == 2
        assert result.num_reasoning_steps == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
