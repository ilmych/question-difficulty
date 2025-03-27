"""
Implementation of parallel processing for the Question Difficulty Framework.

This module provides functions to easily integrate parallel processing
into the existing framework with minimal changes to the main codebase.
"""

import logging
import sys
import os
import time
from typing import List, Dict, Any, Optional, Callable

# Import the parallel processor
from parallel_processor import ParallelProcessor

# Get logger
logger = logging.getLogger("difficulty_framework")

def parallel_assess_difficulty(
    questions: List[Dict[str, Any]],
    client=None,
    max_workers: int = 5,
    show_progress: bool = True
) -> Dict[str, Any]:
    """
    A drop-in replacement for the existing question assessment function
    that uses parallel processing to improve performance.
    
    Args:
        questions: List of question dictionaries to assess
        client: LLM client to use for API calls
        max_workers: Maximum number of parallel workers to use
        show_progress: Whether to show progress bars
        
    Returns:
        Assessment results dictionary with the same structure
        as the sequential version
    """
    from difficulty_framework import (
        aggregate_knowledge_requirements,
        aggregate_cognitive_complexity,
        aggregate_context_and_ambiguity,
        aggregate_question_structure,
        aggregate_linguistic_factors,
        aggregate_learning_objectives,
        calculate_overall_difficulty,
        determine_difficulty_level
    )
    
    # Define all the dimension functions
    dimension_functions = {
        "knowledge_requirements": aggregate_knowledge_requirements,
        "cognitive_complexity": aggregate_cognitive_complexity,
        "context_and_ambiguity": aggregate_context_and_ambiguity,
        "question_structure": aggregate_question_structure,
        "linguistic_factors": aggregate_linguistic_factors,
        "learning_objectives": aggregate_learning_objectives
    }
    
    start_time = time.time()
    logger.info(f"Starting parallel assessment of {len(questions)} questions")
    
    # Process all dimensions in parallel
    dimension_results = ParallelProcessor.parallel_dimension_evaluation(
        questions=questions,
        dimension_functions=dimension_functions,
        client=client,
        max_workers_per_dimension=max_workers,
        sequential_dimensions=True,  # Process dimensions sequentially but questions in parallel
        show_progress=show_progress
    )
    
    # Combine results for each question
    all_results = []
    for i, question in enumerate(questions):
        try:
            # Extract dimension scores for this question
            dimension_scores = {}
            for dimension, results in dimension_results.items():
                if i < len(results):  # Check bounds
                    # Extract score from dimension result
                    dimension_score = results[i][0]  # Assuming [score, details] format
                    dimension_scores[dimension] = dimension_score
            
            # Calculate overall score
            overall_score = calculate_overall_difficulty(dimension_scores)
            difficulty_level = determine_difficulty_level(overall_score)
            
            # Store results for this question
            question_result = {
                "question_text": question["question"],
                "dimension_scores": dimension_scores,
                "overall_score": overall_score,
                "difficulty_level": difficulty_level,
                "details": {
                    # Store detailed results for each dimension
                    "knowledge_requirements": dimension_results["knowledge_requirements"][i][1],
                    "cognitive_complexity": dimension_results["cognitive_complexity"][i][1],
                    "context_and_ambiguity": dimension_results["context_and_ambiguity"][i][1],
                    "question_structure": dimension_results["question_structure"][i][1],
                    "linguistic_factors": dimension_results["linguistic_factors"][i][1],
                    "learning_objectives": dimension_results["learning_objectives"][i][1]
                }
            }
            all_results.append(question_result)
            
        except Exception as e:
            logger.error(f"Error combining results for question {i}: {str(e)}")
            # Add a placeholder result
            all_results.append({
                "question_text": question.get("question", f"Question {i}"),
                "dimension_scores": {},
                "overall_score": 0,
                "difficulty_level": "Unknown",
                "error": str(e)
            })
    
    # Calculate overall test difficulty
    import statistics
    overall_scores = [r["overall_score"] for r in all_results if "overall_score" in r]
    avg_difficulty = statistics.mean(overall_scores) if overall_scores else 0
    overall_difficulty_level = determine_difficulty_level(avg_difficulty)
    
    # Calculate average scores for each dimension
    dimension_averages = {}
    for dimension in dimension_functions.keys():
        scores = [r["dimension_scores"].get(dimension, 0) for r in all_results 
                 if "dimension_scores" in r and dimension in r["dimension_scores"]]
        dimension_averages[dimension] = statistics.mean(scores) if scores else 0
    
    # Prepare final results
    elapsed_time = time.time() - start_time
    logger.info(f"Parallel assessment completed in {elapsed_time:.2f} seconds")
    
    final_results = {
        "questions": all_results,
        "overall_test_difficulty": {
            "score": avg_difficulty,
            "level": overall_difficulty_level,
            "dimension_averages": dimension_averages
        },
        "processing_summary": {
            "total_questions": len(questions),
            "processed": len(all_results),
            "elapsed_time_seconds": elapsed_time,
            "parallel_workers": max_workers
        }
    }
    
    return final_results

# Example usage in main function for direct testing
def main():
    import argparse
    import json
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description="Parallel assessment of question difficulty")
    parser.add_argument("input_file", help="JSON file containing questions")
    parser.add_argument("--output", default="results.json", help="Output file for results")
    parser.add_argument("--workers", type=int, default=5, help="Number of parallel workers")
    parser.add_argument("--no-progress", action="store_true", help="Hide progress bars")
    args = parser.parse_args()
    
    try:
        # Load questions
        with open(args.input_file, 'r') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            questions = data
        elif isinstance(data, dict) and "questions" in data:
            questions = data["questions"]
        else:
            raise ValueError("Invalid question data format")
        
        logger.info(f"Loaded {len(questions)} questions from {args.input_file}")
        
        # Initialize the LLM client
        from difficulty_framework import initialize_anthropic_client
        client = initialize_anthropic_client()
        
        # Process questions in parallel
        results = parallel_assess_difficulty(
            questions=questions, 
            client=client,
            max_workers=args.workers,
            show_progress=not args.no_progress
        )
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {args.output}")
        
        # Print summary
        print(f"\nProcessed {len(questions)} questions in {results['processing_summary']['elapsed_time_seconds']:.2f} seconds")
        print(f"Overall difficulty: {results['overall_test_difficulty']['level']} ({results['overall_test_difficulty']['score']:.2f})")
        
        # Count by difficulty level
        difficulty_counts = {}
        for question in results["questions"]:
            level = question["difficulty_level"]
            difficulty_counts[level] = difficulty_counts.get(level, 0) + 1
        
        print("\nDifficulty distribution:")
        for level, count in sorted(difficulty_counts.items()):
            percentage = (count / len(questions)) * 100
            print(f"  {level}: {count} questions ({percentage:.1f}%)")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()