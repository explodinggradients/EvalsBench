#!/usr/bin/env python3
"""
Benchmark script for evaluating LLM judges on Q&A pairs.

Usage:
    python benchmark.py --csv my_qa_pairs_100.csv --provider openai --model gpt-4o --judges vanilla,fixed_few_shot
"""

import argparse
import asyncio
import json
import os
import typing as t
from pathlib import Path

import instructor
import pandas as pd
from openai import AsyncOpenAI
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,  
    cohen_kappa_score, matthews_corrcoef, confusion_matrix
)
from tqdm import tqdm

from ragas_experimental.embeddings import OpenAIEmbeddings
from ragas_experimental.llms import BaseRagasLLM
from ragas_experimental.llms.llm import InstructorLLM
from ragas_experimental.metrics import DiscreteMetric, Metric
from ragas_experimental.prompt import DynamicFewShotPrompt
from ragas_experimental.prompt.random_few_shot import RandomFewShotPrompt


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark LLM judges on Q&A pairs")
    parser.add_argument("--csv", required=True, help="Path to the CSV file with Q&A pairs")
    parser.add_argument("--provider", required=True, choices=["openai", "anthropic", "gemini"], 
                       help="LLM provider")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--judges", help="Comma-separated list of judges (vanilla,fixed_few_shot,random_few_shot,dynamic_few_shot)")
    parser.add_argument("--output-dir", default="results", help="Output directory for results")
    parser.add_argument("--num-examples", type=int, default=3, help="Number of examples for few-shot judges")
    parser.add_argument("--num-samples", type=int, required=False, help="Number of samples to process from the CSV")
    parser.add_argument("--annotated-samples", type=str, required=True, 
                        help="Path to the JSON file with annotated training samples")
    
    return parser.parse_args()


def get_llm(provider: str, model: str) -> BaseRagasLLM:
    """Get the LLM based on the provider and model."""
    
    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI provider")
        client = AsyncOpenAI(api_key=api_key)
        client = instructor.from_openai(client=client)
    elif provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required for Anthropic provider")
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic(api_key)
        client = instructor.from_anthropic(client=client)
    elif provider == "gemini":
        from google import genai
        client = genai.Client()
        client = instructor.from_genai(client=client, use_async=True)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    llm = InstructorLLM(client=client, model=model)
    return llm


PROMPT = "Check if the response contains points mentioned from the grading notes and return 'pass' or 'fail'.\nResponse: {response} Grading Notes: {grading_notes}"


def vanilla_judge():
    """Create a vanilla judge with no examples."""
    return DiscreteMetric(
        name="vanilla_correctness",
        prompt=PROMPT,
        allowed_values=["pass", "fail"],
    )


def fixed_few_shot_judge(examples: t.List[dict], num_examples: int = 3):
    """Create a judge with fixed few-shot examples."""
    my_metric = DiscreteMetric(
        name="fixed_few_shot_correctness",
        prompt=PROMPT,
        allowed_values=["pass", "fail"],
    )
    for example in examples[:num_examples]:
        my_metric.prompt.add_example(
            inputs=example["input"],
            output=example["output"]    
        ) 
    return my_metric


def random_few_shot_judge(examples: t.List[dict]):
    """Create a judge with random few-shot examples."""
    my_metric = DiscreteMetric(
        name="random_few_shot_correctness",
        prompt=PROMPT,
        allowed_values=["pass", "fail"],
    )
    my_metric.prompt = RandomFewShotPrompt.from_prompt(my_metric.prompt, num_examples=3)
    for example in examples:
        my_metric.prompt.add_example(
            inputs=example["input"],
            output=example["output"]
        )
    return my_metric


def dynamic_few_shot_judge(examples: t.List[dict]):
    """Create a judge with dynamic few-shot examples based on embeddings."""
    my_metric = DiscreteMetric(
        name="dynamic_few_shot_correctness",
        prompt=PROMPT,
        allowed_values=["pass", "fail"],
    )
    embedding = OpenAIEmbeddings(model="text-embedding-3-small", client=AsyncOpenAI())
    my_metric.prompt = DynamicFewShotPrompt.from_prompt(my_metric.prompt, num_examples=3, embedding_model=embedding)
    for example in examples:
        my_metric.prompt.add_example(
            inputs=example["input"],
            output=example["output"]
        )
    return my_metric


def get_judge(metric_type: str, examples: t.List[dict], num_examples: int = 3):
    """Get the judge based on the metric type."""
    if metric_type == "vanilla":
        return vanilla_judge()
    elif metric_type == "fixed_few_shot":
        return fixed_few_shot_judge(examples, num_examples)
    elif metric_type == "random_few_shot":
        return random_few_shot_judge(examples)
    elif metric_type == "dynamic_few_shot":
        return dynamic_few_shot_judge(examples)
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")


async def score_df(df: pd.DataFrame, llm: BaseRagasLLM, metric: Metric) -> pd.DataFrame:
    """Score a dataframe using the given LLM and metric."""
    result_df = df.copy()
    result_df['verdict'] = None
    result_df['reason'] = None
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Scoring with {metric.name}"):
        response = row['response']
        grading_notes = row['grading_notes']
        
        score = await metric.ascore(
            llm=llm,
            response=response,
            grading_notes=grading_notes
        )
        
        result_df.at[idx, 'verdict'] = score.value
        result_df.at[idx, 'reason'] = score.reason
    
    return result_df


def calculate_metrics(y_true, y_pred):
    """Calculate and return all evaluation metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label="pass")
    prec, rec, f1 = float(prec), float(rec), float(f1)
    kappa = float(cohen_kappa_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred).tolist()
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "cohen_kappa": kappa,
        "matthews_corrcoef": mcc,
        "confusion_matrix": cm
    }


async def score_and_save(df: pd.DataFrame, llm: BaseRagasLLM, metric: Metric, output_dir: str) -> None:
    """Score dataframe and save results and metrics."""
    result_df = await score_df(df, llm, metric)
    
    provider = llm.client.provider.name if hasattr(llm.client, 'provider') else 'unknown'
    model = llm.model
    
    # Create output directories
    output_path = Path(output_dir) / provider / model
    output_path.mkdir(parents=True, exist_ok=True)
    
    file_name = f"{metric.name}"
    
    # Save results CSV
    result_df.to_csv(output_path / f"{file_name}.csv", index=False)
    
    # Calculate and save metrics
    y_true = result_df['target'].tolist()
    y_pred = result_df['verdict'].tolist()
    
    metrics = calculate_metrics(y_true, y_pred)
    
    with open(output_path / f"{file_name}.json", 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Results saved to {output_path / file_name}")


async def main():
    """Main function to run the benchmark."""
    args = parse_args()
    
    # Load and transform data
    df = pd.read_csv(args.csv, nrows=args.num_samples if args.num_samples else None)
    annotated_samples = json.load(open(args.annotated_samples, "r"))
    
    # Get LLM
    llm = get_llm(args.provider, args.model)
    
    # Determine which judges to run
    if args.judges:
        judge_types = [j.strip() for j in args.judges.split(',')]
    else:
        judge_types = ["vanilla", "fixed_few_shot", "random_few_shot", "dynamic_few_shot"]
    
    # Run benchmarks for each judge type
    for judge_type in judge_types:
        print(f"\nRunning {judge_type} judge...")
        judge = get_judge(judge_type, annotated_samples, args.num_examples)
        await score_and_save(df, llm, judge, args.output_dir)


if __name__ == "__main__":
    asyncio.run(main())