"""
Progress monitoring for the Question Difficulty Framework.

This module provides utilities for tracking, reporting, and visualizing
progress during question assessment.
"""

import time
import sys
import logging
from typing import Dict, Any, Optional, List, Union, Callable
from tqdm import tqdm

# Get logger
logger = logging.getLogger("difficulty_framework")

class ProgressMonitor:
    """
    Monitor and report progress during question assessment.
    Provides both console feedback and logging of progress information.
    """
    
    def __init__(self, total_questions: int, total_dimensions: int = 6,
                 show_progress_bar: bool = True, log_interval: int = 10):
        """
        Initialize the progress monitor.
        
        Args:
            total_questions: Total number of questions to process
            total_dimensions: Total number of dimensions to evaluate
            show_progress_bar: Whether to show progress bars in the console
            log_interval: How often to log progress (in percent)
        """
        self.total_questions = total_questions
        self.total_dimensions = total_dimensions
        self.show_progress_bar = show_progress_bar
        self.log_interval = log_interval
        
        # Progress tracking
        self.start_time = time.time()
        self.completed_dimensions = 0
        self.completed_questions = 0
        self.current_dimension = None
        
        # Performance metrics
        self.dimension_times = {}
        self.question_processing_times = []
        self.last_progress_time = self.start_time
        self.last_logged_percent = 0
        
        # Error tracking
        self.errors = []
        
        # Create overall progress bar
        if self.show_progress_bar:
            self.overall_progress = tqdm(
                total=self.total_questions * self.total_dimensions,
                desc="Total Progress",
                position=0,
                unit="evaluations"
            )
            
            # Create dimension progress bar
            self.dimension_progress = tqdm(
                total=self.total_questions,
                desc="Current Dimension",
                position=1,
                unit="questions",
                leave=False
            )
        else:
            self.overall_progress = None
            self.dimension_progress = None
            
        # Log initial status
        logger.info(f"Starting assessment of {total_questions} questions across {total_dimensions} dimensions")
    
    def start_dimension(self, dimension_name: str):
        """
        Start processing a new dimension.
        
        Args:
            dimension_name: Name of the dimension being processed
        """
        self.current_dimension = dimension_name
        self.dimension_times[dimension_name] = {
            'start_time': time.time(),
            'end_time': None,
            'elapsed': None
        }
        
        # Reset dimension progress bar
        if self.dimension_progress:
            self.dimension_progress.reset()
            self.dimension_progress.set_description(f"Dimension: {dimension_name}")
        
        logger.info(f"Starting dimension: {dimension_name}")
    
    def complete_dimension(self, dimension_name: str):
        """
        Mark a dimension as completed.
        
        Args:
            dimension_name: Name of the completed dimension
        """
        if dimension_name in self.dimension_times:
            end_time = time.time()
            elapsed = end_time - self.dimension_times[dimension_name]['start_time']
            
            self.dimension_times[dimension_name]['end_time'] = end_time
            self.dimension_times[dimension_name]['elapsed'] = elapsed
            
            self.completed_dimensions += 1
            
            # Log completion
            logger.info(f"Completed dimension: {dimension_name} in {elapsed:.2f} seconds")
            
            # Update overall progress to reflect completion of the full dimension
            if self.overall_progress:
                # Ensure we don't double-count if question completions were already tracked
                remaining = self.total_questions - self.overall_progress.n % self.total_questions
                if remaining > 0:
                    self.overall_progress.update(remaining)
    
    def complete_question(self, dimension_name: str, question_index: int, 
                         success: bool = True, error: Optional[str] = None):
        """
        Mark a question as completed for the current dimension.
        
        Args:
            dimension_name: Name of the current dimension
            question_index: Index of the completed question
            success: Whether the question was processed successfully
            error: Error message if processing failed
        """
        # Track completion
        self.completed_questions += 1
        
        # Track error if any
        if not success and error:
            self.errors.append({
                'dimension': dimension_name,
                'question_index': question_index,
                'error': error
            })
        
        # Update progress bars
        if self.dimension_progress:
            self.dimension_progress.update(1)
        
        if self.overall_progress:
            self.overall_progress.update(1)
        
        # Log progress at intervals
        current_time = time.time()
        elapsed = current_time - self.start_time
        progress_percent = (self.completed_questions / (self.total_questions * self.total_dimensions)) * 100
        
        # Log at specified intervals or if we have an error
        if (int(progress_percent) // self.log_interval > 
            self.last_logged_percent // self.log_interval) or not success:
            
            questions_per_second = self.completed_questions / max(elapsed, 0.001)
            estimated_total = (self.total_questions * self.total_dimensions) / max(questions_per_second, 0.001)
            remaining = max(estimated_total - elapsed, 0)
            
            log_msg = (
                f"Progress: {progress_percent:.1f}% ({self.completed_questions}/{self.total_questions * self.total_dimensions}) "
                f"- {questions_per_second:.2f} q/s - Est. remaining: {remaining:.1f}s"
            )
            
            if not success:
                logger.warning(f"{log_msg} - Error in dimension {dimension_name}, question {question_index}: {error}")
            else:
                logger.info(log_msg)
            
            self.last_logged_percent = int(progress_percent)
            self.last_progress_time = current_time
    
    def finalize(self) -> Dict[str, Any]:
        """
        Finalize progress monitoring and return summary statistics.
        
        Returns:
            Dictionary with progress summary and statistics
        """
        end_time = time.time()
        total_elapsed = end_time - self.start_time
        
        # Close progress bars
        if self.overall_progress:
            self.overall_progress.close()
        if self.dimension_progress:
            self.dimension_progress.close()
        
        # Prepare summary
        summary = {
            'total_questions': self.total_questions,
            'total_dimensions': self.total_dimensions,
            'completed_questions': self.completed_questions,
            'completed_dimensions': self.completed_dimensions,
            'success_rate': (self.completed_questions - len(self.errors)) / max(self.completed_questions, 1) * 100,
            'error_count': len(self.errors),
            'total_elapsed_seconds': total_elapsed,
            'questions_per_second': self.completed_questions / max(total_elapsed, 0.001),
            'dimension_times': self.dimension_times
        }
        
        # Log summary
        logger.info(f"Assessment completed in {total_elapsed:.2f} seconds")
        logger.info(f"Processed {self.completed_questions} question evaluations ({summary['questions_per_second']:.2f} per second)")
        
        if self.errors:
            logger.warning(f"Encountered {len(self.errors)} errors during processing")
        
        # Log dimension times
        for dim, times in self.dimension_times.items():
            if times['elapsed'] is not None:
                logger.info(f"Dimension {dim}: {times['elapsed']:.2f} seconds")
        
        return summary
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finalize()
        return False  # Don't suppress exceptions

# Convenience function to create a simple progress bar
def create_progress_bar(total: int, description: str = "Processing", 
                       unit: str = "items", position: int = 0) -> tqdm:
    """
    Create a tqdm progress bar with consistent formatting.
    
    Args:
        total: Total number of items to process
        description: Description text for the progress bar
        unit: Unit label for the items
        position: Position of the progress bar (for multiple bars)
        
    Returns:
        Configured tqdm progress bar
    """
    return tqdm(
        total=total,
        desc=description,
        unit=unit,
        position=position,
        leave=True if position == 0 else False,
        bar_format="{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )

def log_progress(
    current: int, total: int, 
    description: str = "Progress", 
    log_interval: int = 10,
    logger: logging.Logger = logger,
    timer_start: Optional[float] = None
) -> None:
    """
    Log progress at specified intervals without a progress bar.
    Useful for environments where tqdm can't be used.
    
    Args:
        current: Current progress value
        total: Total items to process
        description: Description for the log message
        log_interval: How often to log (in percent)
        logger: Logger to use for logging messages
        timer_start: Optional start time for calculating speed and ETA
    """
    if total <= 0:
        return
        
    progress_percent = (current / total) * 100
    
    # Log at specified intervals
    if progress_percent % log_interval < (log_interval / total) * 100:
        msg = f"{description}: {current}/{total} ({progress_percent:.1f}%)"
        
        # Add timing information if timer_start is provided
        if timer_start is not None:
            elapsed = time.time() - timer_start
            rate = current / max(elapsed, 0.001)
            eta = (total - current) / max(rate, 0.001)
            
            msg += f" - {rate:.2f} items/s - ETA: {eta:.1f}s"
            
        logger.info(msg)

# Example of how to use the ProgressMonitor with a simple task
def example_with_progress():
    """Example function showing how to use the ProgressMonitor."""
    # 100 questions, 6 dimensions
    questions = [{"id": i} for i in range(100)]
    dimensions = ["knowledge", "cognitive", "context", "structure", "linguistic", "objectives"]
    
    # Initialize progress monitor
    with ProgressMonitor(len(questions), len(dimensions)) as monitor:
        
        # Process each dimension
        for dimension in dimensions:
            monitor.start_dimension(dimension)
            
            # Process each question for this dimension
            for i, question in enumerate(questions):
                # Simulate work
                time.sleep(0.01)
                
                # Simulate occasional errors
                success = i % 20 != 0
                error = "Simulated error" if not success else None
                
                # Mark question as completed
                monitor.complete_question(dimension, i, success, error)
            
            # Mark dimension as completed
            monitor.complete_dimension(dimension)
    
    print("Example completed!")

if __name__ == "__main__":
    # Configure logging for the example
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the example
    example_with_progress()