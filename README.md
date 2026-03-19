# Reasoning-Oriented Prompting and Decoding Strategies in LLMs

A comprehensive benchmark comparing prompting techniques (Direct, CoT, ToT, ToG, Reflexion) and decoding strategies (Greedy, Beam Search, Top-k/Top-p) on reasoning tasks.

## Project Structure

```
CSE_495B-NLP/
├── src/
│   ├── prompting/          # Prompting strategy implementations
│   ├── decoding/           # Decoding strategy implementations
│   ├── models/             # Model loading and configuration
│   ├── evaluation/         # Metrics and evaluation framework
│   └── utils/              # Utility functions
├── data/
│   ├── arithmetic/         # GSM8K and math problems
│   ├── logic/              # LogiQA and logic puzzles
│   └── reading/            # Reading comprehension datasets
├── experiments/            # Experiment configurations and scripts
├── results/                # Output results and analysis
├── notebooks/              # Jupyter notebooks for analysis
└── tests/                  # Unit tests
```

## Installation

### Local Setup (Mac/Linux)
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Google Colab Free Setup
```bash
# Open COLAB.md and run the notebook cells there
# See COLAB.md for complete setup and run steps
```

## Quick Start

```bash
# Run a single experiment
python -m experiments.run_experiment --model qwen2.5-3b --prompting cot --decoding greedy --dataset gsm8k

# Run full benchmark
python -m experiments.run_experiment --config experiments/configs/full_benchmark.yaml

# Colab-friendly benchmark preset
python -m experiments.run_experiment --config experiments/configs/colab_free.yaml
```

## Prompting Strategies

| Strategy | Description |
|----------|-------------|
| **Direct** | Standard prompting without reasoning guidance |
| **Chain-of-Thought (CoT)** | Step-by-step reasoning with "Let's think step by step" |
| **Tree-of-Thought (ToT)** | Explores multiple reasoning paths with self-evaluation |
| **Tree-of-Graph (ToG)** | Graph-based reasoning with path merging and contradiction resolution |
| **Reflexion** | Iterative self-refinement through feedback |

## Decoding Strategies

| Strategy | Key Parameters |
|----------|----------------|
| **Greedy** | `do_sample=False` |
| **Beam Search** | `num_beams=5` |
| **Top-k Sampling** | `top_k=50, temperature=0.7` |
| **Top-p (Nucleus)** | `top_p=0.9, temperature=0.7` |

## Evaluation Metrics

- **Accuracy**: Correctness of final answers
- **Reasoning Trace Length**: Number of reasoning steps
- **Hallucination Rate**: Factual inconsistencies in reasoning
- **Latency**: Time per inference

## Supported Models

- Qwen2.5 (3B)
- LLaMA-2 (7B, 13B)
- Mistral (7B)
- Mixtral (8x7B)

## Citation

```bibtex
@misc{reasoning-prompting-2026,
  title={A Comparative Study of Reasoning-Oriented Prompting and Decoding Strategies},
  author={Your Name},
  year={2026}
}
```

## References

- Wei et al. (2022), "Chain of Thought Prompting Elicits Reasoning in Large Language Models"
- Yao et al. (2023), "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"
- Mialon et al. (2023), "Self-Refinement Through Feedback"
