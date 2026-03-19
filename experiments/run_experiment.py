"""
Main Experiment Runner

Runs experiments combining different prompting and decoding strategies.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from itertools import product
import logging
from dataclasses import dataclass, asdict
from tqdm import tqdm
import yaml

import torch

# Import project modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import ModelLoader
from src.prompting import get_prompting_strategy
from src.decoding import get_decoding_strategy
from src.data import get_dataset
from src.evaluation import ReasoningEvaluator
from src.utils import set_seed, setup_logging, get_device_info

# Optional: GPU optimizations for Colab/cloud
try:
    from src.utils.gpu_utils import optimize_for_gpu, clear_gpu_cache
    GPU_UTILS_AVAILABLE = True
except ImportError:
    GPU_UTILS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    model_name: str
    prompting_strategy: str
    decoding_strategy: str
    dataset_name: str
    
    # Model settings
    quantization: Optional[str] = "4bit"
    
    # Data settings
    max_samples: Optional[int] = None
    split: str = "test"
    
    # Generation settings
    max_new_tokens: int = 512
    
    # Decoding-specific settings
    num_beams: int = 5
    top_k: int = 50
    top_p: float = 0.9
    temperature: float = 0.7
    
    # Prompting-specific settings
    use_few_shot: bool = False
    tot_num_branches: int = 3
    tot_max_depth: int = 3
    reflexion_max_iterations: int = 3
    tog_num_hypotheses: int = 3
    tog_max_depth: int = 4
    tog_exploration_width: int = 2
    
    # Experiment settings
    seed: int = 42
    output_dir: str = "results"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExperimentRunner:
    """
    Main experiment runner that orchestrates the entire benchmark.
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
    ):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.tokenizer = None
        self.prompting_strategy = None
        self.decoding_strategy = None
        self.dataset = None

        # Apply GPU optimizations if available.
        if GPU_UTILS_AVAILABLE and torch.cuda.is_available():
            logger.info("Applying GPU optimizations...")
            optimize_for_gpu()
        
        self.evaluator = None
    
    def setup(self) -> None:
        """Initialize all components."""
        logger.info("Setting up experiment...")
        set_seed(self.config.seed)
        
        # Log device info
        device_info = get_device_info()
        logger.info(f"Device info: {device_info}")
        
        # Load model
        logger.info(f"Loading model: {self.config.model_name}")
        model_loader = ModelLoader(
            model_name=self.config.model_name,
            quantization=self.config.quantization,
        )
        self.model, self.tokenizer = model_loader.load()
        
        # Setup decoding strategy
        logger.info(f"Setting up decoding: {self.config.decoding_strategy}")
        self.decoding_strategy = self._create_decoding_strategy()
        
        # Setup prompting strategy
        logger.info(f"Setting up prompting: {self.config.prompting_strategy}")
        self.prompting_strategy = self._create_prompting_strategy()
        
        # Load dataset
        logger.info(f"Loading dataset: {self.config.dataset_name}")
        self.dataset = get_dataset(
            self.config.dataset_name,
            split=self.config.split,
            max_samples=self.config.max_samples,
            seed=self.config.seed,
        )
        self.dataset.load()
        
        # Setup evaluator
        self.evaluator = ReasoningEvaluator(
            task_type=self.dataset.task_type,
            output_dir=str(self.output_dir),
        )
        
        logger.info("Setup complete!")
    
    def _create_decoding_strategy(self):
        """Create decoding strategy based on config."""
        strategy_name = self.config.decoding_strategy
        
        if strategy_name == "greedy":
            return get_decoding_strategy(
                "greedy",
                max_new_tokens=self.config.max_new_tokens,
            )
        elif strategy_name == "beam_search":
            return get_decoding_strategy(
                "beam_search",
                num_beams=self.config.num_beams,
                max_new_tokens=self.config.max_new_tokens,
            )
        elif strategy_name == "top_k":
            return get_decoding_strategy(
                "top_k",
                top_k=self.config.top_k,
                temperature=self.config.temperature,
                max_new_tokens=self.config.max_new_tokens,
            )
        elif strategy_name == "top_p":
            return get_decoding_strategy(
                "top_p",
                top_p=self.config.top_p,
                temperature=self.config.temperature,
                max_new_tokens=self.config.max_new_tokens,
            )
        elif strategy_name == "combined":
            return get_decoding_strategy(
                "combined",
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                temperature=self.config.temperature,
                max_new_tokens=self.config.max_new_tokens,
            )
        else:
            raise ValueError(f"Unknown decoding strategy: {strategy_name}")
    
    def _create_prompting_strategy(self):
        """Create prompting strategy based on config."""
        strategy_name = self.config.prompting_strategy
        generation_config = self.decoding_strategy.get_generation_kwargs()
        
        if strategy_name == "direct":
            return get_prompting_strategy(
                "direct",
                model=self.model,
                tokenizer=self.tokenizer,
                generation_config=generation_config,
            )
        elif strategy_name == "cot":
            return get_prompting_strategy(
                "cot",
                model=self.model,
                tokenizer=self.tokenizer,
                generation_config=generation_config,
                use_few_shot=self.config.use_few_shot,
            )
        elif strategy_name == "tot":
            return get_prompting_strategy(
                "tot",
                model=self.model,
                tokenizer=self.tokenizer,
                generation_config=generation_config,
                num_branches=self.config.tot_num_branches,
                max_depth=self.config.tot_max_depth,
            )
        elif strategy_name == "reflexion":
            return get_prompting_strategy(
                "reflexion",
                model=self.model,
                tokenizer=self.tokenizer,
                generation_config=generation_config,
                max_iterations=self.config.reflexion_max_iterations,
            )
        elif strategy_name == "tog":
            return get_prompting_strategy(
                "tog",
                model=self.model,
                tokenizer=self.tokenizer,
                generation_config=generation_config,
                num_initial_thoughts=self.config.tog_num_hypotheses,
                max_depth=self.config.tog_max_depth,
                exploration_width=self.config.tog_exploration_width,
            )
        else:
            raise ValueError(f"Unknown prompting strategy: {strategy_name}")
    
    def run(self) -> Dict[str, Any]:
        """Run the experiment."""
        logger.info("Starting experiment run...")
        start_time = time.time()
        
        predictions = []
        
        # Process each sample
        for sample in tqdm(self.dataset, desc="Processing samples"):
            try:
                result = self.prompting_strategy.generate(
                    question=sample.question,
                    context=sample.context,
                )
                
                predictions.append({
                    "final_answer": result.final_answer,
                    "reasoning_trace": result.reasoning_trace,
                    "raw_output": result.raw_output,
                    "num_reasoning_steps": result.num_reasoning_steps,
                    "metadata": result.metadata,
                })
            except Exception as e:
                logger.error(f"Error processing sample {sample.idx}: {e}")

                # CUDA device-side assert corrupts the current CUDA context.
                # Stop early so users can restart runtime and avoid misleading results.
                if "device-side assert triggered" in str(e).lower():
                    raise RuntimeError(
                        "CUDA device-side assert triggered during generation. "
                        "Please restart the Colab runtime and rerun with safer decoding settings "
                        "(e.g., greedy or lower-temperature sampling)."
                    ) from e

                predictions.append({
                    "final_answer": "",
                    "reasoning_trace": [],
                    "raw_output": f"ERROR: {str(e)}",
                    "num_reasoning_steps": 0,
                    "metadata": {"error": str(e)},
                })
        
        # Evaluate
        references = self.dataset.get_references()
        experiment_name = f"{self.config.model_name}_{self.config.prompting_strategy}_{self.config.decoding_strategy}_{self.config.dataset_name}"
        
        eval_result = self.evaluator.evaluate(
            predictions=predictions,
            references=references,
            experiment_name=experiment_name,
        )
        
        # Prepare results
        total_time = time.time() - start_time
        results = {
            "config": self.config.to_dict(),
            "evaluation": {
                "accuracy": eval_result.accuracy,
                "exact_match": eval_result.exact_match,
                "reasoning_metrics": eval_result.reasoning_metrics,
                "hallucination_metrics": eval_result.hallucination_metrics,
            },
            "runtime": {
                "total_seconds": total_time,
                "samples_per_second": len(self.dataset) / total_time,
            },
        }
        
        # Save results
        self._save_results(results, predictions, experiment_name)
        
        logger.info(f"Experiment complete in {total_time:.2f}s")
        logger.info(f"Accuracy: {eval_result.accuracy:.2%}")
        
        return results
    
    def _save_results(
        self,
        results: Dict[str, Any],
        predictions: List[Dict[str, Any]],
        experiment_name: str
    ) -> None:
        """Save experiment results to disk."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save summary
        summary_path = self.output_dir / f"{experiment_name}_{timestamp}_summary.json"
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        
        # Save detailed predictions
        predictions_path = self.output_dir / f"{experiment_name}_{timestamp}_predictions.json"
        with open(predictions_path, "w") as f:
            json.dump(predictions, f, indent=2)
        
        logger.info(f"Results saved to {self.output_dir}")


def run_benchmark(config_path: str) -> Dict[str, Any]:
    """
    Run full benchmark from config file.
    
    Config file should specify lists of models, prompting strategies,
    decoding strategies, and datasets to test all combinations.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    models = config.get("models", ["qwen2.5-3b"])
    prompting_strategies = config.get("prompting_strategies", ["direct", "cot"])
    decoding_strategies = config.get("decoding_strategies", ["greedy", "top_p"])
    datasets = config.get("datasets", ["gsm8k"])
    
    all_results = {}
    
    # Run all combinations
    combinations = list(product(models, prompting_strategies, decoding_strategies, datasets))
    logger.info(f"Running {len(combinations)} experiment combinations")
    
    for model, prompting, decoding, dataset in combinations:
        exp_name = f"{model}_{prompting}_{decoding}_{dataset}"
        logger.info(f"\n{'='*60}\nRunning: {exp_name}\n{'='*60}")
        
        try:
            exp_config = ExperimentConfig(
                model_name=model,
                prompting_strategy=prompting,
                decoding_strategy=decoding,
                dataset_name=dataset,
                **{k: v for k, v in config.items() 
                   if k not in ["models", "prompting_strategies", "decoding_strategies", "datasets"]}
            )
            
            runner = ExperimentRunner(exp_config)
            runner.setup()
            results = runner.run()
            all_results[exp_name] = results

            if GPU_UTILS_AVAILABLE and torch.cuda.is_available():
                clear_gpu_cache()
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Experiment {exp_name} failed: {e}")
            all_results[exp_name] = {"error": str(e)}
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run reasoning benchmark experiments")
    
    # Single experiment args
    parser.add_argument("--model", type=str, default="qwen2.5-3b", help="Model name")
    parser.add_argument("--prompting", type=str, default="cot", help="Prompting strategy")
    parser.add_argument("--decoding", type=str, default="greedy", help="Decoding strategy")
    parser.add_argument("--dataset", type=str, default="gsm8k", help="Dataset name")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to evaluate")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Benchmark mode
    parser.add_argument("--config", type=str, default=None, help="Config file for full benchmark")
    
    # Logging
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=getattr(logging, args.log_level))
    
    if args.config:
        # Run full benchmark
        results = run_benchmark(args.config)
    else:
        # Run single experiment
        config = ExperimentConfig(
            model_name=args.model,
            prompting_strategy=args.prompting,
            decoding_strategy=args.decoding,
            dataset_name=args.dataset,
            max_samples=args.max_samples,
            output_dir=args.output_dir,
            seed=args.seed,
        )
        
        runner = ExperimentRunner(config)
        runner.setup()
        results = runner.run()
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
