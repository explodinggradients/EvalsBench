# EvalsBench

A comprehensive benchmark suite for evaluating the effectiveness of Large Language Models (LLMs) as judges in Q&A assessment tasks. This project compares different prompting strategies and evaluates their performance across multiple providers and models.

## TL;DR

| Provider  | Model                | Vanilla | Fixed Few Shot | Random Few Shot | Dynamic Few Shot | Automatic Prompt Optimisation |
|-----------|----------------------|---------|----------------|-----------------|------------------|-------------------------------|
| Anthropic | claude-3.5-haiku     | 66.95   | 67.8           | 66.95           | 66.1             | 74.42                         |
| Anthropic | claude-4-opus        | 86.02   | 94.34          | 91.14           | 95.18            | 95.12                         |
| Anthropic | claude-4-sonnet      | 87.43   | 92.12          | 90.12           | 90.57            | 93.57                         |
| Google    | gemini-2.5-flash     | 87.91   | 87.78          | 90.4            | 89.89            | 98.16                         |
| Google    | gemini-2.5-flash-lite| 89.53   | 80.77          | 76.83           | 79.49            | 56.64                         |
| Google    | gemini-2.5-pro       | 86.19   | 92.4           | 94.05           | 92.94            | 95.71                         |
| OpenAI    | gpt-4o               | 80.63   | 82.35          | 80.63           | 80.0             | 92.86                         |
| OpenAI    | gpt-4o-mini          | 84.49   | 79.79          | 75.86           | 76.65            | 51.38                         |
| OpenAI    | o3-mini              | 87.43   | 95.76          | 93.02           | 91.43            | 95.81                         |

## 🎯 Overview

This project implements four different judge types to evaluate LLM performance in grading Q&A pairs:

- **Vanilla Judge**: Basic prompting without examples
- **Fixed Few-Shot Judge**: Uses a fixed set of examples for context
- **Random Few-Shot Judge**: Randomly samples examples for each evaluation
- **Dynamic Few-Shot Judge**: Uses embedding similarity to select relevant examples
- **Optimised Judge**: Applies a post-processing step to improve verdicts


### Environment Setup

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Set up your API keys as environment variables:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
# GOOGLE_API_KEY may be required for Gemini
```

### Run the Benchmark

Run a benchmark with the default settings:

```bash
python src/benchmark.py --csv data/benchmark_df.csv --provider anthropic --model claude-sonnet-4-20250514  --annotated-samples data/annotation_df.csv  --num-examples 2 
```

### Create Scoreboard
Run the scoreboard script to generate a summary of the results:

```bash
python create_scoreboard.py --output my_scores.csv --decimal-places 2
```


## 🙏 Acknowledgments

- Built with [Ragas](https://github.com/explodinggradients/ragas)
