"""
Results Analysis and Visualization

Tools for analyzing and visualizing benchmark results.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


def load_results(results_dir: str) -> Dict[str, Any]:
    """Load all experiment results from a directory."""
    results_path = Path(results_dir)
    results = {}
    
    for file_path in results_path.glob("*_summary.json"):
        with open(file_path) as f:
            data = json.load(f)
            exp_name = file_path.stem.replace("_summary", "")
            results[exp_name] = data
    
    return results


def results_to_dataframe(results: Dict[str, Any]) -> pd.DataFrame:
    """Convert results to a pandas DataFrame for analysis."""
    rows = []
    
    for exp_name, data in results.items():
        if "error" in data:
            continue
        
        config = data.get("config", {})
        evaluation = data.get("evaluation", {})
        
        row = {
            "experiment": exp_name,
            "model": config.get("model_name", "unknown"),
            "prompting": config.get("prompting_strategy", "unknown"),
            "decoding": config.get("decoding_strategy", "unknown"),
            "dataset": config.get("dataset_name", "unknown"),
            "accuracy": evaluation.get("accuracy", 0),
            "exact_match": evaluation.get("exact_match", 0),
            "avg_reasoning_steps": evaluation.get("reasoning_metrics", {}).get("avg_reasoning_steps", 0),
            "hallucination_rate": evaluation.get("hallucination_metrics", {}).get("contradiction_rate", 0),
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def plot_accuracy_comparison(df: pd.DataFrame, output_path: Optional[str] = None):
    """Plot accuracy comparison across prompting and decoding strategies."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy by prompting strategy
    prompting_acc = df.groupby("prompting")["accuracy"].mean().sort_values(ascending=False)
    axes[0].bar(prompting_acc.index, prompting_acc.values, color='steelblue')
    axes[0].set_xlabel("Prompting Strategy")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Average Accuracy by Prompting Strategy")
    axes[0].set_ylim(0, 1)
    
    # Accuracy by decoding strategy
    decoding_acc = df.groupby("decoding")["accuracy"].mean().sort_values(ascending=False)
    axes[1].bar(decoding_acc.index, decoding_acc.values, color='coral')
    axes[1].set_xlabel("Decoding Strategy")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Average Accuracy by Decoding Strategy")
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_heatmap(df: pd.DataFrame, output_path: Optional[str] = None):
    """Plot heatmap of prompting × decoding accuracy."""
    pivot = df.pivot_table(
        values="accuracy",
        index="prompting",
        columns="decoding",
        aggfunc="mean"
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2%",
        cmap="YlGnBu",
        ax=ax,
        vmin=0,
        vmax=1,
    )
    ax.set_title("Accuracy: Prompting × Decoding Strategies")
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_reasoning_vs_accuracy(df: pd.DataFrame, output_path: Optional[str] = None):
    """Plot relationship between reasoning steps and accuracy."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    scatter = ax.scatter(
        df["avg_reasoning_steps"],
        df["accuracy"],
        c=df["prompting"].astype("category").cat.codes,
        s=100,
        alpha=0.7,
    )
    
    ax.set_xlabel("Average Reasoning Steps")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reasoning Depth vs. Accuracy")
    
    # Add legend
    prompting_strategies = df["prompting"].unique()
    handles = [plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=plt.cm.tab10(i), markersize=10)
               for i in range(len(prompting_strategies))]
    ax.legend(handles, prompting_strategies, title="Prompting")
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig


def generate_report(results_dir: str, output_dir: str) -> str:
    """Generate a comprehensive analysis report."""
    results = load_results(results_dir)
    df = results_to_dataframe(results)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    plot_accuracy_comparison(df, output_path / "accuracy_comparison.png")
    plot_heatmap(df, output_path / "accuracy_heatmap.png")
    plot_reasoning_vs_accuracy(df, output_path / "reasoning_vs_accuracy.png")
    
    # Generate text report
    report_lines = [
        "# Benchmark Results Report",
        "",
        "## Summary Statistics",
        "",
        f"- Total experiments: {len(df)}",
        f"- Models tested: {df['model'].nunique()}",
        f"- Datasets: {df['dataset'].unique().tolist()}",
        "",
        "## Overall Accuracy",
        "",
        f"- Mean accuracy: {df['accuracy'].mean():.2%}",
        f"- Best accuracy: {df['accuracy'].max():.2%}",
        f"- Worst accuracy: {df['accuracy'].min():.2%}",
        "",
        "## Best Configurations",
        "",
    ]
    
    # Top 5 configurations
    top5 = df.nlargest(5, "accuracy")[["experiment", "accuracy", "prompting", "decoding"]]
    for _, row in top5.iterrows():
        report_lines.append(
            f"- {row['prompting']} + {row['decoding']}: {row['accuracy']:.2%}"
        )
    
    report_lines.extend([
        "",
        "## Prompting Strategy Comparison",
        "",
    ])
    
    for prompting in df["prompting"].unique():
        acc = df[df["prompting"] == prompting]["accuracy"].mean()
        report_lines.append(f"- {prompting}: {acc:.2%}")
    
    report_lines.extend([
        "",
        "## Decoding Strategy Comparison",
        "",
    ])
    
    for decoding in df["decoding"].unique():
        acc = df[df["decoding"] == decoding]["accuracy"].mean()
        report_lines.append(f"- {decoding}: {acc:.2%}")
    
    report_text = "\n".join(report_lines)
    
    # Save report
    with open(output_path / "report.md", "w") as f:
        f.write(report_text)
    
    # Save dataframe
    df.to_csv(output_path / "results.csv", index=False)
    
    return report_text


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("--results-dir", type=str, required=True, help="Results directory")
    parser.add_argument("--output-dir", type=str, default="analysis", help="Output directory")
    
    args = parser.parse_args()
    
    report = generate_report(args.results_dir, args.output_dir)
    print(report)
