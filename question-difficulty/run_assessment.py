#!/usr/bin/env python3
"""
Launcher script for the Question Difficulty Framework with cache support.
"""

import sys
import os
import logging
import argparse
import json
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the current directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the framework
from difficulty_framework import initialize_anthropic_client
from parallel_implementation import parallel_assess_difficulty

def main():
    """Run question difficulty assessment with cache control."""
    parser = argparse.ArgumentParser(description="Question Difficulty Assessment Framework")
    parser.add_argument("input_file", help="JSON file containing questions")
    parser.add_argument("--output", default="results.json", help="Output file for results")
    parser.add_argument("--workers", type=int, default=5, help="Number of parallel workers")
    parser.add_argument("--no-progress", action="store_true", help="Hide progress bars")
    
    # Cache-related arguments
    parser.add_argument("--cache-dir", help="Directory to store cache files")
    parser.add_argument("--ttl", type=int, default=30, help="Cache time-to-live in days")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--clear-cache", action="store_true", help="Clear the cache before running")
    parser.add_argument("--cache-stats", action="store_true", help="Print cache statistics after running")
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Initialize cache if the module is available
    try:
        from cache_manager import get_cache
        
        cache = get_cache(
            cache_dir=args.cache_dir,
            ttl_days=args.ttl,
            enabled=not args.no_cache
        )
        
        # Clear cache if requested
        if args.clear_cache and cache.enabled:
            cleared = cache.clear_all()
            logger.info(f"Cleared {cleared} cache entries")
        
        cache_enabled = cache.enabled
    except ImportError:
        logger.warning("Cache manager not available, continuing without caching")
        cache_enabled = False
    
    try:
        # Load questions
        with open(args.input_file, 'r', encoding='utf-8') as f:
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
        client = initialize_anthropic_client()
        
        # Process questions in parallel with cache support
        results = parallel_assess_difficulty(
            questions=questions, 
            client=client,
            max_workers=args.workers,
            show_progress=not args.no_progress,
            use_cache=cache_enabled
        )
        
        # Save results
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {args.output}")
        
        # Print summary
        elapsed_time = time.time() - start_time
        print(f"\nProcessed {len(questions)} questions in {elapsed_time:.2f} seconds")
        print(f"Average time per question: {elapsed_time / len(questions):.2f} seconds")
        print(f"Overall difficulty: {results['overall_test_difficulty']['level']} ({results['overall_test_difficulty']['score']:.2f})")
        
        # Print cache stats if requested
        if args.cache_stats and cache_enabled:
            stats = cache.get_stats()
            print("\nCache Statistics:")
            print(f"  Hits: {stats['hits']}, Misses: {stats['misses']}")
            print(f"  Hit Rate: {stats['hit_rate']:.1f}%")
            print(f"  Errors: {stats['errors']}")
        
        # Count by difficulty level
        difficulty_counts = {}
        for question in results["questions"]:
            level = question["difficulty_level"]
            difficulty_counts[level] = difficulty_counts.get(level, 0) + 1
        
        print("\nDifficulty distribution:")
        for level, count in sorted(difficulty_counts.items()):
            percentage = (count / len(questions)) * 100
            print(f"  {level}: {count} questions ({percentage:.1f}%)")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
