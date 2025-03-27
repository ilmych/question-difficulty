"""
Parallel processing functionality for the Question Difficulty Framework.

This module provides utilities for parallel evaluation of questions
using multithreading to improve performance.
"""

import concurrent.futures
import time
import logging
from typing import List, Dict, Any, Callable, Optional
from tqdm import tqdm

# Get logger
logger = logging.getLogger("difficulty_framework")

class ParallelProcessor:
    """
    Processes multiple questions in parallel using multithreading.
    """
    
    @staticmethod
    def process_questions(
        questions: List[Dict[str, Any]],
        processing_function: Callable,
        max_workers: int = 5,
        description: str = "Processing questions",
        rate_limit: float = 0.5,  # Minimum seconds between API calls
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process multiple questions in parallel using ThreadPoolExecutor.
        
        Args:
            questions: List of question dictionaries to process
            processing_function: Function that processes a single question
                                 Should accept a question dict and return a result dict
            max_workers: Maximum number of parallel threads to use
            description: Description for the progress bar
            rate_limit: Minimum seconds between API calls to prevent rate limiting
            show_progress: Whether to show a progress bar
            
        Returns:
            List of processed results in the same order as the input questions
        """
        results = []
        errors = []
        
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create future to question mapping to maintain order
            future_to_question = {}
            
            # Submit all tasks
            for i, question in enumerate(questions):
                # Add a small delay between submissions to prevent rate limiting
                if i > 0 and rate_limit > 0:
                    time.sleep(rate_limit)
                
                # Submit the task
                future = executor.submit(
                    ParallelProcessor._safely_process_question,
                    processing_function, question, i
                )
                future_to_question[future] = i
            
            # Create progress bar if requested
            if show_progress:
                progress_iter = tqdm(
                    concurrent.futures.as_completed(future_to_question),
                    total=len(questions),
                    desc=description
                )
            else:
                progress_iter = concurrent.futures.as_completed(future_to_question)
            
            # Process results as they complete
            for future in progress_iter:
                question_idx = future_to_question[future]
                try:
                    result, error = future.result()
                    
                    # Store result
                    results.append({
                        'question_idx': question_idx,
                        'result': result
                    })
                    
                    # Store error if any
                    if error:
                        errors.append({
                            'question_idx': question_idx,
                            'error': error
                        })
                        
                except Exception as e:
                    logger.error(f"Error processing question {question_idx}: {str(e)}")
                    errors.append({
                        'question_idx': question_idx,
                        'error': str(e)
                    })
                    results.append({
                        'question_idx': question_idx,
                        'result': None
                    })
        
        # Sort results by question index to maintain original order
        results.sort(key=lambda x: x['question_idx'])
        
        # Log error summary if any errors occurred
        if errors:
            logger.warning(f"Completed with {len(errors)} errors out of {len(questions)} questions")
        else:
            logger.info(f"Successfully processed all {len(questions)} questions")
        
        # Return just the results, not the indices
        return [r['result'] for r in results]
    
    @staticmethod
    def _safely_process_question(
        processing_function: Callable,
        question: Dict[str, Any],
        index: int
    ) -> tuple:
        """
        Safely process a single question with error handling.
        
        Args:
            processing_function: Function to process the question
            question: Question dictionary to process
            index: Index of the question in the original list
            
        Returns:
            Tuple of (result, error)
        """
        error = None
        result = None
        
        try:
            result = processing_function(question)
        except Exception as e:
            error = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Error processing question {index}: {error}")
        
        return result, error
    
    @staticmethod
    def process_dimension(
        questions: List[Dict[str, Any]],
        dimension_function: Callable,
        dimension_name: str,
        client=None,
        max_workers: int = 5,
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process a specific dimension for multiple questions in parallel.
        
        Args:
            questions: List of question dictionaries to process
            dimension_function: Function that evaluates a dimension for a single question
                                Should accept question and client parameters
            dimension_name: Name of the dimension being evaluated (for logging)
            client: LLM client to use for evaluation
            max_workers: Maximum number of parallel threads to use
            show_progress: Whether to show a progress bar
            
        Returns:
            List of dimension results in the same order as the input questions
        """
        # Create a processing function that includes the client
        def process_with_client(question):
            return dimension_function(question, client)
        
        # Use the general parallel processor
        return ParallelProcessor.process_questions(
            questions=questions,
            processing_function=process_with_client,
            max_workers=max_workers,
            description=f"Evaluating {dimension_name}",
            show_progress=show_progress
        )
    
    @staticmethod
    def parallel_dimension_evaluation(
        questions: List[Dict[str, Any]],
        dimension_functions: Dict[str, Callable],
        client=None,
        max_workers_per_dimension: int = 3,
        sequential_dimensions: bool = True,
        show_progress: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Evaluate multiple dimensions for multiple questions.
        
        Args:
            questions: List of question dictionaries to evaluate
            dimension_functions: Dictionary mapping dimension names to evaluation functions
            client: LLM client to use for evaluation
            max_workers_per_dimension: Maximum workers to use per dimension
            sequential_dimensions: If True, process dimensions sequentially
                                   If False, process all dimensions in parallel
            show_progress: Whether to show progress bars
            
        Returns:
            Dictionary mapping dimension names to lists of dimension results
        """
        dimension_results = {}
        
        if sequential_dimensions:
            # Process each dimension sequentially, with parallel question processing
            for dimension_name, dimension_function in dimension_functions.items():
                logger.info(f"Processing dimension: {dimension_name}")
                
                results = ParallelProcessor.process_dimension(
                    questions=questions,
                    dimension_function=dimension_function,
                    dimension_name=dimension_name,
                    client=client,
                    max_workers=max_workers_per_dimension,
                    show_progress=show_progress
                )
                
                dimension_results[dimension_name] = results
        else:
            # Advanced mode: Process all dimensions in parallel
            # This uses more resources but is faster
            # Not implemented in this version for simplicity
            logger.warning("Parallel dimension evaluation not implemented, using sequential processing")
            
            # Fall back to sequential processing
            for dimension_name, dimension_function in dimension_functions.items():
                results = ParallelProcessor.process_dimension(
                    questions=questions,
                    dimension_function=dimension_function,
                    dimension_name=dimension_name,
                    client=client,
                    max_workers=max_workers_per_dimension,
                    show_progress=show_progress
                )
                
                dimension_results[dimension_name] = results
        
        return dimension_results