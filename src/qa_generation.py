#!/usr/bin/env python3
"""
QA Generation Pipeline - Command Line Interface

Generates startup/tech domain QA pairs with expert grading criteria.
Extracted from notebooks/data_gen.ipynb for production use.
"""

import asyncio
import argparse
import csv
import os
import sys
from datetime import datetime
from typing import List

import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel


class TopicsResponse(BaseModel):
    events: List[str]


class QAResponse(BaseModel):
    question: str
    grading_notes: str


class AnswerResponse(BaseModel):
    answer: str


class ModifiedAnswerResponse(BaseModel):
    modified_answer: str
    changes_made: str


class QAGenerator:
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.client = instructor.from_openai(AsyncOpenAI(api_key=api_key))

    async def generate_topics(self, num_topics: int) -> List[str]:
        """Generate startup/tech topics for QA generation."""
        prompt = f"""Generate a list of {num_topics} specific topics related to tech bubbles, startups, and the venture capital ecosystem. Each topic should be a focused subject that would require domain expertise to evaluate properly.

Focus on topics that involve:
- Startup valuation methods and metrics
- Venture capital funding rounds and terms
- Tech bubble warning signs and indicators  
- Startup business model evaluation
- Market timing and competitive dynamics
- Regulatory and compliance issues for tech companies
- Financial modeling for startups
- Due diligence processes
- Exit strategies (IPO, acquisition)
- Tech market cycles and patterns

Examples of good topics:
- "Pre-money vs post-money valuation calculations"
- "Series A liquidation preferences"
- "SaaS revenue recognition standards"
- "Burn rate optimization strategies"
- "Market cap to revenue ratios in growth stocks"

Generate topics that require specific knowledge where precision in dates, numbers, percentages, or technical terms would matter significantly for accurate evaluation."""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a venture capital and startup ecosystem expert."},
                    {"role": "user", "content": prompt},
                ],
                response_model=TopicsResponse,
            )
            return response.events
        except Exception as e:
            raise RuntimeError(f"Failed to generate topics: {e}")

    async def generate_qa_pair(self, topic: str) -> tuple[str, str]:
        """Generate question and grading notes for a given topic."""
        prompt = f"""
Generate a question and grading notes for the given startup/tech topic. Write grading notes as if quickly jotted down by a busy domain expert - informal, abbreviated, with key details that matter.

Requirements:
- Question should require specific startup/VC knowledge where precision matters
- Grading notes should sound like quick expert notes: abbreviated phrases, bullet fragments, key numbers/dates
- Include critical details but in shorthand style (like "post-$ val", "LTV:CAC 3:1+", "18mo runway min")
- Mark super critical items with * or ! 
- 25-35 words max, informal expert jargon okay

Examples of expert note style:
- "Series A: $2-15M, *post-money val*, board seats, liquidation prefs, anti-dilution std"
- "SaaS metrics: *ARR growth*, churn <5%, LTV:CAC 3:1+, payback <12mo"
- "Burn rate calc: monthly cash out, exclude one-time, *18mo runway* benchmark"

Format:
Question: [Precision-requiring startup question]
Grading Notes: [Quick expert jottings with key details & abbreviations]

Topic: {topic}
"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a VC partner quickly jotting evaluation criteria between meetings."},
                    {"role": "user", "content": prompt},
                ],
                response_model=QAResponse,
            )
            return response.question, response.grading_notes
        except Exception as e:
            raise RuntimeError(f"Failed to generate QA pair for topic '{topic}': {e}")

    async def generate_complete_answer(self, question: str, grading_notes: str) -> str:
        """Generate a comprehensive answer covering all grading criteria."""
        prompt = f"""
Generate a comprehensive answer that fully addresses the question and covers all points mentioned in the grading notes. Write as a startup/VC domain expert providing actionable, precise information.

Requirements:
- Address every point mentioned in the grading notes
- Include specific numbers, percentages, timeframes where indicated
- Use proper startup/VC terminology
- Provide practical, actionable guidance
- Maintain professional but accessible tone
- Be precise with critical details (marked with * or ! in grading notes)

**Question:** {question}

**Grading Notes:** {grading_notes}

Write a complete answer that an expert would give, covering all grading criteria with appropriate detail and precision.
"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[  
                    {"role": "system", "content": "You are a seasoned venture capital partner and startup advisor providing expert guidance."},
                    {"role": "user", "content": prompt}
                ],
                response_model=AnswerResponse,
            )
            return response.answer
        except Exception as e:
            raise RuntimeError(f"Failed to generate complete answer: {e}")

    async def generate_modified_answer(self, question: str, grading_notes: str, complete_answer: str) -> tuple[str, str]:
        """Generate a modified answer with strategically dropped important points."""
        prompt = f"""
Given the question, grading notes, and original answer, create a modified version that strategically drops 1-2 important points while maintaining overall coherence.

## Instructions:
1. **Target Selection**: Identify 1-2 important points from the grading notes to completely omit from the answer
2. **Prioritize Critical Items**: Focus on dropping items marked with * or ! in grading notes (expert-flagged as critical)
3. **Maintain Flow**: Ensure the answer still reads naturally despite missing information
4. **Preserve Other Details**: Keep all other points from grading notes intact and precise
5. **Add Compensatory Content**: Include other relevant but non-essential details to maintain answer length and plausibility

## Dropping Strategy:
- Remove specific metrics, benchmarks, or requirements ("18 months runway", "3:1+ LTV:CAC ratio")
- Omit critical process steps or considerations ("exclude one-time expenses", "board seats")
- Drop important qualifications or warnings ("anti-dilution provisions", "regulatory requirements")
- Skip key technical specifications ("post-money valuation", "liquidation preferences")

## What to Maintain:
- Overall answer structure and professional tone
- Non-dropped points with full precision
- Natural transitions between remaining points
- Sufficient detail to seem comprehensive

**Question:** {question}
**Grading Notes:** {grading_notes}
**Original Answer:** {complete_answer}

Generate a version that drops 1-2 important points from the grading notes while maintaining the appearance of being complete and authoritative.
"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[  
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                response_model=ModifiedAnswerResponse,
            )
            return response.modified_answer, response.changes_made
        except Exception as e:
            raise RuntimeError(f"Failed to generate modified answer: {e}")

    async def generate_qa_dataset(self, num_samples: int) -> List[dict]:
        """Generate complete QA dataset with all components."""
        print(f"Generating {num_samples} QA pairs...")
        
        # Generate topics
        print("Generating topics...")
        topics = await self.generate_topics(num_samples)
        
        qa_data = []
        for i, topic in enumerate(topics, 1):
            print(f"Processing topic {i}/{num_samples}: {topic[:50]}...")
            
            try:
                # Generate QA pair
                question, grading_notes = await self.generate_qa_pair(topic)
                
                # Generate complete answer
                complete_answer = await self.generate_complete_answer(question, grading_notes)
                
                # Generate modified answer
                modified_answer, changes_made = await self.generate_modified_answer(
                    question, grading_notes, complete_answer
                )
                
                qa_data.append({
                    'topic': topic,
                    'question': question,
                    'grading_notes': grading_notes,
                    'complete_answer': complete_answer,
                    'modified_answer': modified_answer,
                    'changes_made': changes_made
                })
                
            except Exception as e:
                print(f"Error processing topic {i}: {e}")
                continue
        
        return qa_data


def save_to_csv(data: List[dict], output_path: str):
    """Save QA data to CSV file."""
    if not data:
        raise ValueError("No data to save")
    
    fieldnames = ['topic', 'question', 'grading_notes', 'complete_answer', 'modified_answer', 'changes_made']
    
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        print(f"Saved {len(data)} QA pairs to {output_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to save CSV: {e}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate startup/tech domain QA pairs with expert grading criteria",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python qa_generation.py --num 5
  python qa_generation.py --num 20 --model gpt-4o-mini
  python qa_generation.py --num 10 --output my_qa_pairs.csv
        """
    )
    
    parser.add_argument(
        '--num',
        type=int,
        default=10,
        help='Number of QA pairs to generate (default: 10)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o',
        help='OpenAI model to use (default: gpt-4o)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output CSV file path (default: qa_pairs_YYYYMMDD_HHMMSS.csv)'
    )
    
    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()
    
    # Validate arguments
    if args.num <= 0:
        print("Error: --num must be positive")
        sys.exit(1)
    
    # Set default output path if not provided
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"qa_pairs_{timestamp}.csv"
    
    try:
        # Initialize generator
        generator = QAGenerator(model=args.model)
        
        # Generate QA dataset
        qa_data = await generator.generate_qa_dataset(args.num)
        
        if not qa_data:
            print("Error: No QA pairs were generated successfully")
            sys.exit(1)
        
        # Save to CSV
        save_to_csv(qa_data, args.output)
        
        print(f"Successfully generated {len(qa_data)} QA pairs!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())