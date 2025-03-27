# analyze_passages.py
from sat_passage_analyzer import SATPassageAnalyzer
# Import our parallelized functions
from parallelized_analyzer import *
import time
import os
import logging
logger = logging.getLogger("difficulty_framework")

def main():
    start_time = time.time()
    
    # Initialize the analyzer
    analyzer = SATPassageAnalyzer()
    
    # Load data
    passages = analyzer.load_data("wic-gen-passages.json", file_type="json")
    logger.info(f"Loaded {len(analyzer.passages)} passages")
    
    # Set your Anthropic API key
    analyzer.init_anthropic_client(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    # Set parameters for processing
    max_workers = 10  # Start with a conservative number to avoid hitting rate limits
    max_retries = 5  # Number of retry attempts for each metric
    retry_delay = 10  # Wait time between retries in seconds
    
    # Calculate readability metrics (non-API dependent first)
    logger.info("=== Calculating Readability Metrics ===")
    fk = analyzer.calculate_flesch_kincaid()
    asl = analyzer.calculate_avg_sentence_length()
    
    # Calculate vocabulary metrics (non-API dependent)
    logger.info("\n=== Calculating Vocabulary Metrics ===")
    try:
        vocab_difficulty = analyzer.calculate_vocabulary_difficulty_ratio()
        academic_usage = analyzer.calculate_academic_word_usage()
    except Exception as e:
        logger.error(f"Error calculating vocabulary metrics: {e}", exc_info=True)
    
    # Now use the parallel functions for the API-dependent metrics
    
    # Lexile API
    try:
        lexile = analyzer.run_with_retry(
            analyzer.calculate_lexile_scores_parallel,
            metric_category="readability",
            metric_name="lexile_score",
            max_workers=max_workers,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
    except Exception as e:
        logger.error(f"Error calculating Lexile scores: {e}", exc_info=True)

    # Claude API metrics
    
    # Domain-specific terminology
    domain_specific = analyzer.run_with_retry(
        analyzer.calculate_domain_specific_terminology_parallel,
        metric_category="vocabulary",
        metric_name="domain_specific_terminology",
        max_workers=max_workers,
        max_retries=max_retries,
        retry_delay=retry_delay
    )
    
    # Syntactic complexity
    subordinate = analyzer.run_with_retry(
        analyzer.calculate_subordinate_clauses_parallel,
        metric_category="syntactic_complexity",
        metric_name="subordinate_clauses",
        max_workers=max_workers,
        max_retries=max_retries,
        retry_delay=retry_delay
    )
    
    syntax_variety = analyzer.run_with_retry(
        analyzer.calculate_syntactic_variety_parallel,
        metric_category="syntactic_complexity",
        metric_name="syntactic_variety",
        max_workers=max_workers,
        max_retries=max_retries,
        retry_delay=retry_delay
    )
    
    inversions = analyzer.run_with_retry(
        analyzer.calculate_structural_inversions_parallel,
        metric_category="syntactic_complexity",
        metric_name="structural_inversions",
        max_workers=max_workers,
        max_retries=max_retries,
        retry_delay=retry_delay
    )
    
    embedded = analyzer.run_with_retry(
        analyzer.calculate_embedded_clauses_parallel,
        metric_category="syntactic_complexity",
        metric_name="embedded_clauses",
        max_workers=max_workers,
        max_retries=max_retries,
        retry_delay=retry_delay
    )

    abstraction = analyzer.run_with_retry(
        analyzer.calculate_abstraction_level_parallel,
        metric_category="conceptual_density",
        metric_name="abstraction_level",
        max_workers=max_workers,
        max_retries=max_retries,
        retry_delay=retry_delay
    )
    
    familiarity = analyzer.run_with_retry(
        analyzer.calculate_concept_familiarity_parallel,
        metric_category="conceptual_density",
        metric_name="concept_familiarity",
        max_workers=max_workers,
        max_retries=max_retries,
        retry_delay=retry_delay
    )

    implied = analyzer.run_with_retry(
        analyzer.calculate_implied_information_parallel,
        metric_category="conceptual_density",
        metric_name="implied_information",
        max_workers=max_workers,
        max_retries=max_retries,
        retry_delay=retry_delay
    )

    argumentative = analyzer.run_with_retry(
        analyzer.calculate_argumentative_complexity_parallel,
        metric_category="rhetorical_structure",
        metric_name="argumentative_complexity",
        max_workers=max_workers,
        max_retries=max_retries,
        retry_delay=retry_delay
    )

    organizational = analyzer.run_with_retry(
        analyzer.calculate_organizational_clarity_parallel,
        metric_category="rhetorical_structure",
        metric_name="organizational_clarity",
        max_workers=max_workers,
        max_retries=max_retries,
        retry_delay=retry_delay
    )

    transitional = analyzer.run_with_retry(
        analyzer.calculate_transitional_elements_parallel,
        metric_category="rhetorical_structure",
        metric_name="transitional_elements",
        max_workers=max_workers,
        max_retries=max_retries,
        retry_delay=retry_delay
    )

    prior_knowledge = analyzer.run_with_retry(
        analyzer.calculate_prior_knowledge_requirements_parallel,
        metric_category="content_accessibility",
        metric_name="prior_knowledge_requirements",
        max_workers=max_workers,
        max_retries=max_retries,
        retry_delay=retry_delay
    )

    disciplinary = analyzer.run_with_retry(
        analyzer.calculate_disciplinary_perspective_parallel,
        metric_category="content_accessibility",
        metric_name="disciplinary_perspective",
        max_workers=max_workers,
        max_retries=max_retries,
        retry_delay=retry_delay
    )

    language_modernity = analyzer.run_with_retry(
        analyzer.calculate_language_modernity_parallel,
        metric_category="content_accessibility",
        metric_name="language_modernity",
        max_workers=max_workers,
        max_retries=max_retries,
        retry_delay=retry_delay
    )

    inference = analyzer.run_with_retry(
        analyzer.calculate_inference_requirement_parallel,
        metric_category="cognitive_demands",
        metric_name="inference_requirement",
        max_workers=max_workers,
        max_retries=max_retries,
        retry_delay=retry_delay
    )

    figurative = analyzer.run_with_retry(
        analyzer.calculate_figurative_language_parallel,
        metric_category="cognitive_demands",
        metric_name="figurative_language",
        max_workers=max_workers,
        max_retries=max_retries,
        retry_delay=retry_delay
    )

    authors = analyzer.run_with_retry(
        analyzer.calculate_authors_purpose_parallel,
        metric_category="cognitive_demands",
        metric_name="authors_purpose",
        max_workers=max_workers,
        max_retries=max_retries,
        retry_delay=retry_delay
    )
    
    # Calculate overall difficulty
    logger.info("\n=== Calculating Overall Difficulty Scores ===")
    overall_scores = analyzer.calculate_overall_difficulty()
    
    # Generate reports
    report_paths = analyzer.generate_report(output_dir="results-wic-generated")
    
    end_time = time.time()
    logger.info(f"Analysis complete! Reports saved to: {report_paths}")
    logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()