#!/usr/bin/env python3
"""
Create scoreboard CSV from results directory using F1 scores.

This script scans the results/ directory, extracts F1 scores from JSON files,
and generates a scoreboard CSV matching the sample_scoreboard.csv format.

Usage:
    python create_scoreboard.py
    python create_scoreboard.py --results-dir results --output scoreboard.csv
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate scoreboard CSV from results")
    parser.add_argument("--results-dir", default="results",
                        help="Path to results directory (default: results)")
    parser.add_argument("--output", default="scoreboard.csv",
                        help="Output CSV file path (default: scoreboard.csv)")
    parser.add_argument("--decimal-places", type=int, default=1,
                        help="Number of decimal places for F1 scores "
                             "(default: 1)")
    return parser.parse_args()


def extract_f1_score(json_file: Path) -> Optional[float]:
    """Extract F1 score from a JSON results file."""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('f1_score')
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None


def get_model_display_name(model_dir: str) -> str:
    """Convert directory model name to display name."""
    # Handle special cases for cleaner display names
    model_mapping = {
        'gpt-4o-mini': 'gpt-4o-mini',
        'gpt-4o': 'gpt-4o',
        'o3-mini': 'o3-mini',
        'claude-3-5-haiku-latest': 'claude-3.5-haiku',
        'claude-opus-4-20250514': 'claude-4-opus',
        'claude-sonnet-4-20250514': 'claude-4-sonnet',
        'gemini-2.5-pro': 'gemini-2.5-pro',
        'gemini-2.5-flash': 'gemini-2.5-flash',
        'gemini-2.5-flash-lite': 'gemini-2.5-flash-lite'
    }
    return model_mapping.get(model_dir, model_dir)


def get_provider_display_name(provider_dir: str) -> str:
    """Convert directory provider name to display name."""
    provider_mapping = {
        'OPENAI': 'OpenAI',
        'ANTHROPIC': 'Anthropic',
        'GENAI': 'Google'
    }
    return provider_mapping.get(provider_dir, provider_dir)


def scan_results_directory(results_dir: Path) -> Dict:
    """Scan results directory and extract all F1 scores."""

    # Define the mapping from file names to column names
    judge_column_mapping = {
        'vanilla_correctness.json': 'Vanilla',
        'fixed_few_shot_correctness.json': 'Fixed few shot',
        'random_few_shot_correctness.json': 'Random few shot',
        'dynamic_few_shot_correctness.json': 'Dynamic Few shot',
        'optimised_correctness.json': 'Automatic prompt optimisation'
    }

    scoreboard_data = {}

    # Walk through provider directories
    for provider_path in results_dir.iterdir():
        if not provider_path.is_dir():
            continue

        provider_name = get_provider_display_name(provider_path.name)

        # Walk through model directories
        for model_path in provider_path.iterdir():
            if not model_path.is_dir():
                continue

            model_name = get_model_display_name(model_path.name)
            key = (provider_name, model_name)

            if key not in scoreboard_data:
                scoreboard_data[key] = {}

            # Look for JSON result files
            for json_file_name, column_name in judge_column_mapping.items():
                json_file_path = model_path / json_file_name
                f1_score = extract_f1_score(json_file_path)

                if f1_score is not None:
                    scoreboard_data[key][column_name] = f1_score

    return scoreboard_data


def create_scoreboard_dataframe(scoreboard_data: Dict,
                                decimal_places: int = 1) -> pd.DataFrame:
    """Create pandas DataFrame from scoreboard data."""

    # Define all possible columns in order
    all_columns = [
        'Vanilla',
        'Fixed few shot',
        'Random few shot',
        'Dynamic Few shot',
        'Automatic prompt optimisation'
    ]

    # Prepare data for DataFrame
    rows = []
    for (provider, model), scores in scoreboard_data.items():
        row = {'Provider': provider, 'Model': model}

        # Add scores for each judge type, formatting as needed
        for column in all_columns:
            if column in scores:
                # Format to specified decimal places
                formatted_score = round(scores[column] * 100, decimal_places)
                row[column] = formatted_score
            else:
                row[column] = ''  # Empty string for missing scores

        rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Sort by Provider then Model for consistent ordering
    df = df.sort_values(['Provider', 'Model']).reset_index(drop=True)

    return df


def main():
    """Main function to generate scoreboard."""
    args = parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory '{results_dir}' does not exist.")
        return

    print(f"Scanning results directory: {results_dir}")

    # Scan directory and extract F1 scores
    scoreboard_data = scan_results_directory(results_dir)

    if not scoreboard_data:
        print("No results found in directory.")
        return

    print(f"Found results for {len(scoreboard_data)} model configurations.")

    # Create DataFrame
    df = create_scoreboard_dataframe(scoreboard_data, args.decimal_places)

    # Save to CSV
    output_path = Path(args.output)
    df.to_csv(output_path, index=False)

    print(f"Scoreboard saved to: {output_path}")
    print("\nScoreboard preview:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()