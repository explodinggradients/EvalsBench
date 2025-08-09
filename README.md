# 🤖 LLM Judge Research

A comprehensive benchmark suite for evaluating the effectiveness of Large Language Models (LLMs) as judges in Q&A assessment tasks. This project compares different prompting strategies and evaluates their performance across multiple providers and models.

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