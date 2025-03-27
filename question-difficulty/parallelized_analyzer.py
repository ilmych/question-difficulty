# parallelized_analyzer.py
import concurrent.futures
import time
from functools import partial
from typing import List, Dict, Any, Callable, Optional, Union
import json

from sat_passage_analyzer import SATPassageAnalyzer, extract_json_from_response

def parallel_api_calls(self, function: Callable, max_workers: int = 5) -> List[Dict[str, Any]]:
    """
    Execute a function across all passages in parallel.
    
    Parameters:
    - function: Function to call on each passage (must accept passage_index as argument)
    - max_workers: Maximum number of parallel workers
    
    Returns:
    - List of results from all function calls
    """
    if not self.passages:
        print("Error: No passages loaded. Please load passages first.")
        return []
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a list of future objects
        futures = [executor.submit(function, passage_index=idx) 
                  for idx in range(len(self.passages))]
        
        # Process results as they complete
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                result = future.result()
                if result:
                    results.append(result)
                    print(f"Completed passage {i+1}/{len(futures)}")
            except Exception as e:
                print(f"Error processing passage: {e}")
    
    return results

# Add this function to the SATPassageAnalyzer class
SATPassageAnalyzer.parallel_api_calls = parallel_api_calls

def parallel_process_metric(self, metric_name: str, process_function: Callable, 
                           max_workers: int = 5, system_prompt: Optional[str] = None,
                           missing_indices: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    """
    Generic function to process any metric in parallel.
    
    Parameters:
    - metric_name: Name of the metric to process
    - process_function: Function to process a single passage
    - max_workers: Maximum number of parallel workers
    - system_prompt: Optional custom system prompt
    - missing_indices: Optional list of passage indices to process (if None, processes all)
    
    Returns:
    - List of results
    """
    print(f"\n=== Calculating {metric_name} in Parallel (Max Workers: {max_workers}) ===")
    
    # If specific indices are provided, only process those
    indices_to_process = missing_indices if missing_indices is not None else range(len(self.passages))
    
    if missing_indices is not None:
        print(f"Processing only {len(missing_indices)} passages with missing data")
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a list of future objects for the specified indices
        futures = [executor.submit(process_function, passage_index=idx) 
                  for idx in indices_to_process]
        
        # Process results as they complete
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                result = future.result()
                if result:
                    results.append(result)
                    print(f"Completed {metric_name} for passage {i+1}/{len(futures)}")
            except Exception as e:
                print(f"Error processing {metric_name}: {e}")
    
    self._save_metric(metric_name, results)
    return results

# Update the SATPassageAnalyzer class with the modified function
SATPassageAnalyzer.parallel_process_metric = parallel_process_metric

def calculate_domain_specific_terminology_single(self, passage_index: int, system_prompt: Optional[str] = None) -> Dict[str, Any]:
    """Process a single passage for domain-specific terminology."""
    try:
        passage = self.passages[passage_index]
        if 'text' not in passage:
            print(f"Warning: No 'text' field found for passage {passage.get('id', 'unknown')}")
            return {}
            
        text = passage['text']
        passage_id = passage.get('id', passage.get('passage_id', 'unknown'))
        title = passage.get('title', f"Passage {passage_id}")
        
        if system_prompt is None:
            sys_prompt = "You are an expert linguist specializing in vocabulary analysis. Provide accurate, objective assessments."
        else:
            sys_prompt = system_prompt
            
        PROMPT_TEMPLATE = """
        I need you to analyze the domain-specific terminology in the following passage. 
        Domain-specific terms are specialized vocabulary words that are primarily used within a particular field or discipline.
        
        Passage:
        {passage}
        
        Please perform the following analysis:
        1. Identify all domain-specific terminology in the passage
        2. Specify which domain or field each term belongs to
        3. Rate the overall density of domain-specific terminology on a scale from 1-5, where:
           - 1: Very few domain-specific terms, accessible to general readers
           - 2: Some domain-specific terms, but still mostly accessible
           - 3: Moderate use of domain-specific terminology
           - 4: Frequent use of domain-specific terminology, challenging for non-specialists
           - 5: Heavy use of specialized terminology, likely inaccessible to non-experts
        
        Format your response as JSON with the following structure:
        {{
            "domain_specific_terms": [
                {{"term": "example term", "domain": "example domain"}}
            ],
            "score": 3,
            "explanation": "Brief explanation of your rating"
        }}
        
        Return ONLY valid JSON, with NO additional text before or after.
        """
        
        prompt = PROMPT_TEMPLATE.format(passage=text)
        
        # Add a short delay to avoid rate limiting
        time.sleep(0.1)
        
        response = self.anthropic_client.messages.create(
            model=self.claude_params["model"],
            max_tokens=self.claude_params["max_tokens"],
            temperature=self.claude_params["temperature"],
            system=sys_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        
        result_text = response.content[0].text
        result = extract_json_from_response(result_text)
        
        if result is None:
            raise ValueError("Failed to extract valid JSON from API response.")
            
        score = result.get('score', None)
        terms = result.get('domain_specific_terms', [])
        explanation = result.get('explanation', '')
        
        result_dict = {
            'passage_id': passage_id,
            'title': title,
            'domain_specific_terminology_score': score,
            'domain_specific_terms': terms,
            'explanation': explanation
        }
        
        # Update metrics dictionary
        existing_metrics = next((item for item in self.metrics['vocabulary'] if item.get('passage_id') == passage_id), None)
        if existing_metrics:
            existing_metrics['domain_specific_terminology'] = score
        else:
            self.metrics['vocabulary'].append({
                'passage_id': passage_id,
                'title': title,
                'domain_specific_terminology': score
            })
            
        return result_dict
        
    except Exception as e:
        print(f"Exception when calculating domain-specific terminology for passage index {passage_index}: {e}")
        return {}

# Add this function to the SATPassageAnalyzer class
SATPassageAnalyzer.calculate_domain_specific_terminology_single = calculate_domain_specific_terminology_single

def calculate_domain_specific_terminology_parallel(self, max_workers: int = 5, system_prompt: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Calculate the density of domain-specific terminology for all passages in parallel.
    
    Parameters:
    - max_workers: Maximum number of parallel workers
    - system_prompt: Optional custom system prompt for Claude
    
    Returns:
    - List of dictionaries with domain terminology scores
    """
    if not hasattr(self, 'anthropic_client') or not self.anthropic_client:
        print("Error: Anthropic client not initialized. Call init_anthropic_client() first.")
        return []
    
    return self.parallel_process_metric(
        metric_name="domain_specific_terminology",
        process_function=self.calculate_domain_specific_terminology_single,
        max_workers=max_workers,
        system_prompt=system_prompt
    )

# Add this function to the SATPassageAnalyzer class
SATPassageAnalyzer.calculate_domain_specific_terminology_parallel = calculate_domain_specific_terminology_parallel

def calculate_subordinate_clauses_single(self, passage_index: int, system_prompt: Optional[str] = None) -> Dict[str, Any]:
    """Process a single passage for subordinate clauses."""
    try:
        passage = self.passages[passage_index]
        if 'text' not in passage:
            print(f"Warning: No 'text' field found for passage {passage.get('id', 'unknown')}")
            return {}
            
        text = passage['text']
        passage_id = passage.get('id', passage.get('passage_id', 'unknown'))
        title = passage.get('title', f"Passage {passage_id}")
        
        if system_prompt is None:
            sys_prompt = "You are an expert linguist specializing in syntax analysis. Provide accurate, objective assessments."
        else:
            sys_prompt = system_prompt
            
        PROMPT_TEMPLATE = """
        Analyze the subordinate clauses in the following passage. A subordinate clause (or dependent clause) is a clause that cannot stand alone as a complete sentence because it does not express a complete thought.

        Passage:
        {passage}

        Please perform the following analysis:
        1. Identify all subordinate clauses in the passage
        2. Count the total number of sentences in the passage
        3. Count the total number of subordinate clauses
        4. Calculate the ratio of subordinate clauses to sentences
        5. Rate the syntactic complexity due to subordinate clauses on a scale from 1-5, where:
           - 1: Very few subordinate clauses, mostly simple sentences
           - 2: Some subordinate clauses, but primarily simple structures
           - 3: Moderate use of subordinate clauses
           - 4: Frequent use of subordinate clauses, creating complex sentence structures
           - 5: Heavy use of subordinate clauses, including multiple layers of embedding

        Format your response as JSON with the following structure:
        {{
            "sentence_count": 10,
            "subordinate_clause_count": 15,
            "subordinate_to_sentence_ratio": 1.5,
            "examples": ["example subordinate clause 1", "example subordinate clause 2"],
            "complexity_score": 3,
            "explanation": "Brief explanation of your rating"
        }}

        Return ONLY valid JSON, with NO additional text before or after.
        """
        
        prompt = PROMPT_TEMPLATE.format(passage=text)
        
        # Add a short delay to avoid rate limiting
        time.sleep(0.1)
        
        response = self.anthropic_client.messages.create(
            model=self.claude_params["model"],
            max_tokens=self.claude_params["max_tokens"],
            temperature=self.claude_params["temperature"],
            system=sys_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        
        result_text = response.content[0].text
        result = extract_json_from_response(result_text)
        
        if result is None:
            raise ValueError("Failed to extract valid JSON from API response.")
        
        sentence_count = result.get('sentence_count', 0)
        subordinate_count = result.get('subordinate_clause_count', 0)
        ratio = result.get('subordinate_to_sentence_ratio', 0)
        complexity_score = result.get('complexity_score', 0)
        examples = result.get('examples', [])
        explanation = result.get('explanation', '')
        
        result_dict = {
            'passage_id': passage_id,
            'title': title,
            'sentence_count': sentence_count,
            'subordinate_clause_count': subordinate_count,
            'subordinate_to_sentence_ratio': ratio,
            'subordinate_clause_examples': examples,
            'complexity_score': complexity_score,
            'explanation': explanation
        }
        
        # Update metrics dictionary
        existing_metrics = next((item for item in self.metrics['syntactic_complexity'] if item.get('passage_id') == passage_id), None)
        if existing_metrics:
            existing_metrics['subordinate_clauses'] = complexity_score
            existing_metrics['subordinate_to_sentence_ratio'] = ratio
        else:
            self.metrics['syntactic_complexity'].append({
                'passage_id': passage_id,
                'title': title,
                'subordinate_clauses': complexity_score,
                'subordinate_to_sentence_ratio': ratio
            })
            
        return result_dict
        
    except Exception as e:
        print(f"Exception when calculating subordinate clauses for passage index {passage_index}: {e}")
        return {}

# Add this function to the SATPassageAnalyzer class
SATPassageAnalyzer.calculate_subordinate_clauses_single = calculate_subordinate_clauses_single

def calculate_subordinate_clauses_parallel(self, max_workers: int = 5, system_prompt: Optional[str] = None) -> List[Dict[str, Any]]:
    """Calculate subordinate clauses for all passages in parallel."""
    if not hasattr(self, 'anthropic_client') or not self.anthropic_client:
        print("Error: Anthropic client not initialized. Call init_anthropic_client() first.")
        return []
    
    return self.parallel_process_metric(
        metric_name="subordinate_clauses",
        process_function=self.calculate_subordinate_clauses_single,
        max_workers=max_workers,
        system_prompt=system_prompt
    )

# Add this function to the SATPassageAnalyzer class
SATPassageAnalyzer.calculate_subordinate_clauses_parallel = calculate_subordinate_clauses_parallel

def calculate_syntactic_variety_single(self, passage_index: int, system_prompt: Optional[str] = None) -> Dict[str, Any]:
    """Process a single passage for syntactic variety."""
    try:
        passage = self.passages[passage_index]
        if 'text' not in passage:
            print(f"Warning: No 'text' field found for passage {passage.get('id', 'unknown')}")
            return {}
            
        text = passage['text']
        passage_id = passage.get('id', passage.get('passage_id', 'unknown'))
        title = passage.get('title', f"Passage {passage_id}")
        
        if system_prompt is None:
            sys_prompt = "You are an expert linguist specializing in syntactic analysis. Provide accurate, objective assessments."
        else:
            sys_prompt = system_prompt
            
        PROMPT_TEMPLATE = """
        Analyze the syntactic variety in the following passage. Syntactic variety refers to the diversity of sentence structures used in writing.

        Passage:
        {passage}

        Please perform the following analysis:
        1. Identify the different sentence structure types used (simple, compound, complex, compound-complex)
        2. Count how many of each type occur in the passage
        3. Analyze sentence beginnings (how many different ways sentences start)
        4. Analyze sentence lengths (variety in number of words per sentence)
        5. Rate the overall syntactic variety on a scale from 1-5, where:
           - 1: Very little variety, repetitive structures
           - 2: Some variety, but noticeable patterns of repetition
           - 3: Moderate variety in structures
           - 4: Good variety with diverse sentence types and beginnings
           - 5: Excellent variety with sophisticated and diverse structures
        
        Format your response as JSON with the following structure:
        {{
            "sentence_types": {{
                "simple": 5,
                "compound": 3,
                "complex": 4,
                "compound_complex": 2
            }},
            "unique_sentence_beginnings": 8,
            "sentence_length_range": {{
                "min": 5,
                "max": 25,
                "average": 15
            }},
            "variety_score": 3,
            "examples": [
                {{"type": "complex", "sentence": "example of a complex sentence"}},
                {{"type": "compound", "sentence": "example of a compound sentence"}}
            ],
            "explanation": "Brief explanation of your rating"
        }}
        
        Return ONLY valid JSON, with NO additional text before or after.
        """
        
        prompt = PROMPT_TEMPLATE.format(passage=text)
        
        # Add a short delay to avoid rate limiting
        time.sleep(0.1)
        
        response = self.anthropic_client.messages.create(
            model=self.claude_params["model"],
            max_tokens=self.claude_params["max_tokens"],
            temperature=self.claude_params["temperature"],
            system=sys_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        
        result_text = response.content[0].text
        result = extract_json_from_response(result_text)
        
        if result is None:
            raise ValueError("Failed to extract valid JSON from API response.")
        
        variety_score = result.get('variety_score', 0)
        sentence_types = result.get('sentence_types', {})
        unique_beginnings = result.get('unique_sentence_beginnings', 0)
        sentence_length_range = result.get('sentence_length_range', {})
        examples = result.get('examples', [])
        explanation = result.get('explanation', '')
        
        result_dict = {
            'passage_id': passage_id,
            'title': title,
            'syntactic_variety_score': variety_score,
            'sentence_types': sentence_types,
            'unique_sentence_beginnings': unique_beginnings,
            'sentence_length_range': sentence_length_range,
            'examples': examples,
            'explanation': explanation
        }
        
        # Update metrics dictionary
        existing_metrics = next((item for item in self.metrics['syntactic_complexity'] if item.get('passage_id') == passage_id), None)
        if existing_metrics:
            existing_metrics['syntactic_variety'] = variety_score
        else:
            self.metrics['syntactic_complexity'].append({
                'passage_id': passage_id,
                'title': title,
                'syntactic_variety': variety_score
            })
            
        return result_dict
        
    except Exception as e:
        print(f"Exception when calculating syntactic variety for passage index {passage_index}: {e}")
        return {}

# Add this function to the SATPassageAnalyzer class
SATPassageAnalyzer.calculate_syntactic_variety_single = calculate_syntactic_variety_single

def calculate_syntactic_variety_parallel(self, max_workers: int = 5, system_prompt: Optional[str] = None) -> List[Dict[str, Any]]:
    """Calculate syntactic variety for all passages in parallel."""
    if not hasattr(self, 'anthropic_client') or not self.anthropic_client:
        print("Error: Anthropic client not initialized. Call init_anthropic_client() first.")
        return []
    
    return self.parallel_process_metric(
        metric_name="syntactic_variety",
        process_function=self.calculate_syntactic_variety_single,
        max_workers=max_workers,
        system_prompt=system_prompt
    )

# Add this function to the SATPassageAnalyzer class
SATPassageAnalyzer.calculate_syntactic_variety_parallel = calculate_syntactic_variety_parallel

def calculate_structural_inversions_single(self, passage_index: int, system_prompt: Optional[str] = None) -> Dict[str, Any]:
    """Process a single passage for structural inversions."""
    try:
        passage = self.passages[passage_index]
        if 'text' not in passage:
            print(f"Warning: No 'text' field found for passage {passage.get('id', 'unknown')}")
            return {}
            
        text = passage['text']
        passage_id = passage.get('id', passage.get('passage_id', 'unknown'))
        title = passage.get('title', f"Passage {passage_id}")
        
        if system_prompt is None:
            sys_prompt = "You are an expert linguist specializing in sentence structure analysis. Provide accurate, objective assessments."
        else:
            sys_prompt = system_prompt
            
        PROMPT_TEMPLATE = """
        Analyze the structural inversions in the following passage. Structural inversions are sentences that deviate from the standard subject-verb-object pattern, often for emphasis or stylistic effect.

        Passage:
        {passage}

        Please perform the following analysis:
        1. Identify all instances of structural inversions in the passage
        2. Count the total number of sentences in the passage
        3. Count the total number of structural inversions
        4. Calculate the ratio of inversions to total sentences
        5. Rate the use of structural inversions on a scale from 1-5, where:
           - 1: Very few inversions (0-5%)
           - 2: Some inversions (5-10%)
           - 3: Moderate use of inversions (10-15%)
           - 4: Frequent use of inversions (15-20%)
           - 5: Heavy use of inversions (>20%)
        
        Consider the following types of inversions:
        - Subject-auxiliary inversion ("Never have I seen such a thing")
        - Fronting or topicalization ("Quickly he ran")
        - Subject-verb inversion after place adverbials ("Down the street ran the dogs")
        - There/Here-constructions with inverted word order
        - Negative adverbials triggering inversion
        
        Format your response as JSON with the following structure:
        {{
            "sentence_count": 20,
            "inversion_count": 3,
            "inversion_ratio": 0.15,
            "inversion_examples": [
                {{"type": "subject-auxiliary", "sentence": "Never have I seen such a disaster."}}
            ],
            "inversion_score": 3,
            "explanation": "Brief explanation of your rating"
        }}
        
        Return ONLY valid JSON, with NO additional text before or after.
        """
        
        prompt = PROMPT_TEMPLATE.format(passage=text)
        
        # Add a short delay to avoid rate limiting
        time.sleep(0.1)
        
        response = self.anthropic_client.messages.create(
            model=self.claude_params["model"],
            max_tokens=self.claude_params["max_tokens"],
            temperature=self.claude_params["temperature"],
            system=sys_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        
        result_text = response.content[0].text
        result = extract_json_from_response(result_text)
        
        if result is None:
            raise ValueError("Failed to extract valid JSON from API response.")
        
        sentence_count = result.get('sentence_count', 0)
        inversion_count = result.get('inversion_count', 0)
        inversion_ratio = result.get('inversion_ratio', 0)
        inversion_score = result.get('inversion_score', 0)
        examples = result.get('inversion_examples', [])
        explanation = result.get('explanation', '')
        
        result_dict = {
            'passage_id': passage_id,
            'title': title,
            'sentence_count': sentence_count,
            'inversion_count': inversion_count,
            'inversion_ratio': inversion_ratio,
            'inversion_score': inversion_score,
            'inversion_examples': examples,
            'explanation': explanation
        }
        
        # Update metrics dictionary
        existing_metrics = next((item for item in self.metrics['syntactic_complexity'] if item.get('passage_id') == passage_id), None)
        if existing_metrics:
            existing_metrics['structural_inversions'] = inversion_score
        else:
            self.metrics['syntactic_complexity'].append({
                'passage_id': passage_id,
                'title': title,
                'structural_inversions': inversion_score
            })
            
        return result_dict
        
    except Exception as e:
        print(f"Exception when calculating structural inversions for passage index {passage_index}: {e}")
        return {}

# Add this function to the SATPassageAnalyzer class
SATPassageAnalyzer.calculate_structural_inversions_single = calculate_structural_inversions_single

def calculate_structural_inversions_parallel(self, max_workers: int = 5, system_prompt: Optional[str] = None) -> List[Dict[str, Any]]:
    """Calculate structural inversions for all passages in parallel."""
    if not hasattr(self, 'anthropic_client') or not self.anthropic_client:
        print("Error: Anthropic client not initialized. Call init_anthropic_client() first.")
        return []
    
    return self.parallel_process_metric(
        metric_name="structural_inversions",
        process_function=self.calculate_structural_inversions_single,
        max_workers=max_workers,
        system_prompt=system_prompt
    )

# Add this function to the SATPassageAnalyzer class
SATPassageAnalyzer.calculate_structural_inversions_parallel = calculate_structural_inversions_parallel

def calculate_embedded_clauses_single(self, passage_index: int, system_prompt: Optional[str] = None) -> Dict[str, Any]:
    """Process a single passage for embedded clauses."""
    try:
        passage = self.passages[passage_index]
        if 'text' not in passage:
            print(f"Warning: No 'text' field found for passage {passage.get('id', 'unknown')}")
            return {}
            
        text = passage['text']
        passage_id = passage.get('id', passage.get('passage_id', 'unknown'))
        title = passage.get('title', f"Passage {passage_id}")
        
        if system_prompt is None:
            sys_prompt = "You are an expert linguist specializing in clause analysis. Provide accurate, objective assessments."
        else:
            sys_prompt = system_prompt
            
        PROMPT_TEMPLATE = """
        Analyze the embedded clauses in the following passage. Embedded clauses are clauses that are nested within other clauses, adding complexity to sentence structure.

        Passage:
        {passage}

        Please perform the following analysis:
        1. Identify all instances of embedded clauses in the passage
        2. Count the total number of sentences in the passage
        3. Count the total number of embedded clauses
        4. Calculate the ratio of embedded clauses to sentences
        5. Identify the maximum level of embedding (how deeply clauses are nested)
        6. Rate the complexity due to embedded clauses on a scale from 1-5, where:
           - 1: Very few embedded clauses, simple structures
           - 2: Some embedded clauses, mostly one level deep
           - 3: Moderate use of embedded clauses, some deeper nesting
           - 4: Frequent use of embedded clauses, often multiple levels
           - 5: Heavy use of deeply embedded clauses, creating very complex structures
        
        Format your response as JSON with the following structure:
        {{
            "sentence_count": 15,
            "embedded_clause_count": 20,
            "embedded_to_sentence_ratio": 1.33,
            "max_embedding_depth": 3,
            "examples": [
                {{"depth": 2, "sentence": "The book that I bought, which had excellent reviews, was disappointing."}}
            ],
            "embedding_score": 3,
            "explanation": "Brief explanation of your rating"
        }}
        
        Return ONLY valid JSON, with NO additional text before or after.
        """
        
        prompt = PROMPT_TEMPLATE.format(passage=text)
        
        # Add a short delay to avoid rate limiting
        time.sleep(0.1)
        
        response = self.anthropic_client.messages.create(
            model=self.claude_params["model"],
            max_tokens=self.claude_params["max_tokens"],
            temperature=self.claude_params["temperature"],
            system=sys_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        
        result_text = response.content[0].text
        result = extract_json_from_response(result_text)
        
        if result is None:
            raise ValueError("Failed to extract valid JSON from API response.")
        
        sentence_count = result.get('sentence_count', 0)
        embedded_count = result.get('embedded_clause_count', 0)
        embedding_ratio = result.get('embedded_to_sentence_ratio', 0)
        max_depth = result.get('max_embedding_depth', 0)
        embedding_score = result.get('embedding_score', 0)
        examples = result.get('examples', [])
        explanation = result.get('explanation', '')
        
        result_dict = {
            'passage_id': passage_id,
            'title': title,
            'sentence_count': sentence_count,
            'embedded_clause_count': embedded_count,
            'embedded_to_sentence_ratio': embedding_ratio,
            'max_embedding_depth': max_depth,
            'embedding_score': embedding_score,
            'examples': examples,
            'explanation': explanation
        }
        
        # Update metrics dictionary
        existing_metrics = next((item for item in self.metrics['syntactic_complexity'] if item.get('passage_id') == passage_id), None)
        if existing_metrics:
            existing_metrics['embedded_clauses'] = embedding_score
            existing_metrics['max_embedding_depth'] = max_depth
        else:
            self.metrics['syntactic_complexity'].append({
                'passage_id': passage_id,
                'title': title,
                'embedded_clauses': embedding_score,
                'max_embedding_depth': max_depth
            })
            
        return result_dict
        
    except Exception as e:
        print(f"Exception when calculating embedded clauses for passage index {passage_index}: {e}")
        return {}

# Add this function to the SATPassageAnalyzer class
SATPassageAnalyzer.calculate_embedded_clauses_single = calculate_embedded_clauses_single

def calculate_embedded_clauses_parallel(self, max_workers: int = 5, system_prompt: Optional[str] = None) -> List[Dict[str, Any]]:
    """Calculate embedded clauses for all passages in parallel."""
    if not hasattr(self, 'anthropic_client') or not self.anthropic_client:
        print("Error: Anthropic client not initialized. Call init_anthropic_client() first.")
        return []
    
    return self.parallel_process_metric(
        metric_name="embedded_clauses",
        process_function=self.calculate_embedded_clauses_single,
        max_workers=max_workers,
        system_prompt=system_prompt
    )

# Add this function to the SATPassageAnalyzer class
SATPassageAnalyzer.calculate_embedded_clauses_parallel = calculate_embedded_clauses_parallel

def calculate_abstraction_level_single(self, passage_index: int, system_prompt: Optional[str] = None) -> Dict[str, Any]:
    """Process a single passage for abstraction level."""
    try:
        passage = self.passages[passage_index]
        if 'text' not in passage:
            print(f"Warning: No 'text' field found for passage {passage.get('id', 'unknown')}")
            return {}
            
        text = passage['text']
        passage_id = passage.get('id', passage.get('passage_id', 'unknown'))
        title = passage.get('title', f"Passage {passage_id}")
        
        if system_prompt is None:
            sys_prompt = "You are an expert in cognitive linguistics and reading comprehension. Provide accurate, objective assessments."
        else:
            sys_prompt = system_prompt
            
        PROMPT_TEMPLATE = """
        Analyze the level of abstraction in the following passage. Consider the ratio of concrete concepts (tangible, specific, sensory) versus abstract concepts (theoretical, general, intangible).

        Passage:
        {passage}

        Please perform the following analysis:
        1. Identify the main concepts in the passage and classify them as concrete or abstract
        2. Estimate the ratio of concrete to abstract concepts
        3. Determine the overall abstraction level of the passage
        4. Rate the abstraction level on a scale from 1-5, where:
           - 1: Highly concrete, dominated by tangible concepts and specific examples
           - 2: Primarily concrete with some abstract concepts
           - 3: Balanced mix of concrete and abstract concepts
           - 4: Primarily abstract with some concrete elements
           - 5: Highly abstract, dominated by theoretical concepts with few tangible examples
        
        Format your response as JSON with the following structure:
        {{
            "concrete_concepts": ["example1", "example2"],
            "abstract_concepts": ["example1", "example2"],
            "concrete_to_abstract_ratio": 0.5,
            "abstraction_level_score": 4,
            "explanation": "Brief explanation of your rating"
        }}
        
        Return ONLY valid JSON, with NO additional text before or after.
        """
        
        prompt = PROMPT_TEMPLATE.format(passage=text)
        
        # Add a short delay to avoid rate limiting
        time.sleep(0.1)
        
        response = self.anthropic_client.messages.create(
            model=self.claude_params["model"],
            max_tokens=self.claude_params["max_tokens"],
            temperature=self.claude_params["temperature"],
            system=sys_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        
        result_text = response.content[0].text
        result = extract_json_from_response(result_text)
        
        if result is None:
            raise ValueError("Failed to extract valid JSON from API response.")
        
        concrete_concepts = result.get('concrete_concepts', [])
        abstract_concepts = result.get('abstract_concepts', [])
        ratio = result.get('concrete_to_abstract_ratio', 0)
        abstraction_score = result.get('abstraction_level_score', 0)
        explanation = result.get('explanation', '')
        
        result_dict = {
            'passage_id': passage_id,
            'title': title,
            'concrete_concepts': concrete_concepts,
            'abstract_concepts': abstract_concepts,
            'concrete_to_abstract_ratio': ratio,
            'abstraction_level_score': abstraction_score,
            'explanation': explanation
        }
        
        # Update metrics dictionary
        existing_metrics = next((item for item in self.metrics['conceptual_density'] if item.get('passage_id') == passage_id), None)
        if existing_metrics:
            existing_metrics['abstraction_level'] = abstraction_score
        else:
            self.metrics['conceptual_density'].append({
                'passage_id': passage_id,
                'title': title,
                'abstraction_level': abstraction_score
            })
            
        return result_dict
        
    except Exception as e:
        print(f"Exception when calculating abstraction level for passage index {passage_index}: {e}")
        return {}

# Add this function to the SATPassageAnalyzer class
SATPassageAnalyzer.calculate_abstraction_level_single = calculate_abstraction_level_single

def calculate_abstraction_level_parallel(self, max_workers: int = 5, system_prompt: Optional[str] = None) -> List[Dict[str, Any]]:
    """Calculate abstraction level for all passages in parallel."""
    if not hasattr(self, 'anthropic_client') or not self.anthropic_client:
        print("Error: Anthropic client not initialized. Call init_anthropic_client() first.")
        return []
    
    return self.parallel_process_metric(
        metric_name="abstraction_level",
        process_function=self.calculate_abstraction_level_single,
        max_workers=max_workers,
        system_prompt=system_prompt
    )

# Add this function to the SATPassageAnalyzer class
SATPassageAnalyzer.calculate_abstraction_level_parallel = calculate_abstraction_level_parallel

def calculate_concept_familiarity_single(self, passage_index: int, system_prompt: Optional[str] = None) -> Dict[str, Any]:
    """Process a single passage for concept familiarity."""
    try:
        passage = self.passages[passage_index]
        if 'text' not in passage:
            print(f"Warning: No 'text' field found for passage {passage.get('id', 'unknown')}")
            return {}
            
        text = passage['text']
        passage_id = passage.get('id', passage.get('passage_id', 'unknown'))
        title = passage.get('title', f"Passage {passage_id}")
        
        if system_prompt is None:
            sys_prompt = "You are an expert in educational assessment and cognitive development. Provide accurate, objective assessments for high school students taking the SAT exam."
        else:
            sys_prompt = system_prompt
            
        PROMPT_TEMPLATE = """
        Analyze the familiarity of concepts in the following passage. Consider how common the main ideas would be to typical high school students taking the SAT exam.

        Passage:
        {passage}

        Please perform the following analysis:
        1. Identify the main concepts and ideas in the passage
        2. Evaluate how familiar these concepts would be to typical high school students
        3. Consider whether the concepts are commonly taught in high school curricula or encountered in everyday life
        4. Rate the concept familiarity on a scale from 1-5, where:
           - 1: Very familiar concepts, common knowledge for most high school students
           - 2: Mostly familiar concepts with a few less common ideas
           - 3: Mixed familiarity, some common concepts and some specialized knowledge
           - 4: Mostly unfamiliar concepts requiring specialized knowledge
           - 5: Very unfamiliar concepts requiring extensive background knowledge
        
        Format your response as JSON with the following structure:
        {{
            "main_concepts": ["concept1", "concept2"],
            "familiar_concepts": ["familiar1", "familiar2"],
            "unfamiliar_concepts": ["unfamiliar1", "unfamiliar2"],
            "concept_familiarity_score": 3,
            "explanation": "Brief explanation of your rating"
        }}
        
        Return ONLY valid JSON, with NO additional text before or after.
        """
        
        prompt = PROMPT_TEMPLATE.format(passage=text)
        
        # Add a short delay to avoid rate limiting
        time.sleep(0.1)
        
        response = self.anthropic_client.messages.create(
            model=self.claude_params["model"],
            max_tokens=self.claude_params["max_tokens"],
            temperature=self.claude_params["temperature"],
            system=sys_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        
        result_text = response.content[0].text
        result = extract_json_from_response(result_text)
        
        if result is None:
            raise ValueError("Failed to extract valid JSON from API response.")
        
        main_concepts = result.get('main_concepts', [])
        familiar_concepts = result.get('familiar_concepts', [])
        unfamiliar_concepts = result.get('unfamiliar_concepts', [])
        familiarity_score = result.get('concept_familiarity_score', 0)
        explanation = result.get('explanation', '')
        
        result_dict = {
            'passage_id': passage_id,
            'title': title,
            'main_concepts': main_concepts,
            'familiar_concepts': familiar_concepts,
            'unfamiliar_concepts': unfamiliar_concepts,
            'concept_familiarity_score': familiarity_score,
            'explanation': explanation
        }
        
        # Update metrics dictionary
        existing_metrics = next((item for item in self.metrics['conceptual_density'] if item.get('passage_id') == passage_id), None)
        if existing_metrics:
            existing_metrics['concept_familiarity'] = familiarity_score
        else:
            self.metrics['conceptual_density'].append({
                'passage_id': passage_id,
                'title': title,
                'concept_familiarity': familiarity_score
            })
            
        return result_dict
        
    except Exception as e:
        print(f"Exception when calculating concept familiarity for passage index {passage_index}: {e}")
        return {}

# Add this function to the SATPassageAnalyzer class
SATPassageAnalyzer.calculate_concept_familiarity_single = calculate_concept_familiarity_single

def calculate_concept_familiarity_parallel(self, max_workers: int = 5, system_prompt: Optional[str] = None) -> List[Dict[str, Any]]:
    """Calculate concept familiarity for all passages in parallel."""
    if not hasattr(self, 'anthropic_client') or not self.anthropic_client:
        print("Error: Anthropic client not initialized. Call init_anthropic_client() first.")
        return []
    
    return self.parallel_process_metric(
        metric_name="concept_familiarity",
        process_function=self.calculate_concept_familiarity_single,
        max_workers=max_workers,
        system_prompt=system_prompt
    )

# Add this function to the SATPassageAnalyzer class
SATPassageAnalyzer.calculate_concept_familiarity_parallel = calculate_concept_familiarity_parallel

def calculate_implied_information_single(self, passage_index: int, system_prompt: Optional[str] = None) -> Dict[str, Any]:
    """Process a single passage for implied information."""
    try:
        passage = self.passages[passage_index]
        if 'text' not in passage:
            print(f"Warning: No 'text' field found for passage {passage.get('id', 'unknown')}")
            return {}
            
        text = passage['text']
        passage_id = passage.get('id', passage.get('passage_id', 'unknown'))
        title = passage.get('title', f"Passage {passage_id}")
        
        if system_prompt is None:
            sys_prompt = "You are an expert in critical reading and textual analysis. Provide accurate, objective assessments."
        else:
            sys_prompt = system_prompt
            
        PROMPT_TEMPLATE = """
        Analyze the implied information in the following passage. Implied information refers to content that is not explicitly stated but must be inferred by the reader.

        Passage:
        {passage}

        Please perform the following analysis:
        1. Identify key information that is implied but not explicitly stated
        2. Evaluate how much background knowledge is needed to make these inferences
        3. Determine the types of inference required (e.g., bridging gaps, drawing conclusions, recognizing unstated assumptions)
        4. Rate the amount of implied information on a scale from 1-5, where:
           - 1: Very little implied information, mostly explicit
           - 2: Some implied information, but most key points are stated
           - 3: Moderate amount of implied information
           - 4: Significant amount of implied information
           - 5: Heavy reliance on implied information, requiring extensive inferencing
        
        Format your response as JSON with the following structure:
        {{
            "implied_information": [
                {{"inference": "example inference", "type": "type of inference required"}}
            ],
            "background_knowledge_needed": ["example1", "example2"],
            "implied_information_score": 3,
            "explanation": "Brief explanation of your rating"
        }}
        
        Return ONLY valid JSON, with NO additional text before or after.
        """
        
        prompt = PROMPT_TEMPLATE.format(passage=text)
        
        # Add a short delay to avoid rate limiting
        time.sleep(0.1)
        
        response = self.anthropic_client.messages.create(
            model=self.claude_params["model"],
            max_tokens=self.claude_params["max_tokens"],
            temperature=self.claude_params["temperature"],
            system=sys_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        
        result_text = response.content[0].text
        result = extract_json_from_response(result_text)
        
        if result is None:
            raise ValueError("Failed to extract valid JSON from API response.")
        
        implied_information = result.get('implied_information', [])
        background_knowledge = result.get('background_knowledge_needed', [])
        implied_score = result.get('implied_information_score', 0)
        explanation = result.get('explanation', '')
        
        result_dict = {
            'passage_id': passage_id,
            'title': title,
            'implied_information': implied_information,
            'background_knowledge_needed': background_knowledge,
            'implied_information_score': implied_score,
            'explanation': explanation
        }
        
        # Update metrics dictionary
        existing_metrics = next((item for item in self.metrics['conceptual_density'] if item.get('passage_id') == passage_id), None)
        if existing_metrics:
            existing_metrics['implied_information'] = implied_score
        else:
            self.metrics['conceptual_density'].append({
                'passage_id': passage_id,
                'title': title,
                'implied_information': implied_score
            })
            
        return result_dict
        
    except Exception as e:
        print(f"Exception when calculating implied information for passage index {passage_index}: {e}")
        return {}

# Add this function to the SATPassageAnalyzer class
SATPassageAnalyzer.calculate_implied_information_single = calculate_implied_information_single

def calculate_implied_information_parallel(self, max_workers: int = 5, system_prompt: Optional[str] = None) -> List[Dict[str, Any]]:
    """Calculate implied information for all passages in parallel."""
    if not hasattr(self, 'anthropic_client') or not self.anthropic_client:
        print("Error: Anthropic client not initialized. Call init_anthropic_client() first.")
        return []
    
    return self.parallel_process_metric(
        metric_name="implied_information",
        process_function=self.calculate_implied_information_single,
        max_workers=max_workers,
        system_prompt=system_prompt
    )

# Add this function to the SATPassageAnalyzer class
SATPassageAnalyzer.calculate_implied_information_parallel = calculate_implied_information_parallel

def calculate_argumentative_complexity_single(self, passage_index: int, system_prompt: Optional[str] = None) -> Dict[str, Any]:
    """Process a single passage for argumentative complexity."""
    try:
        passage = self.passages[passage_index]
        if 'text' not in passage:
            print(f"Warning: No 'text' field found for passage {passage.get('id', 'unknown')}")
            return {}
            
        text = passage['text']
        passage_id = passage.get('id', passage.get('passage_id', 'unknown'))
        title = passage.get('title', f"Passage {passage_id}")
        
        if system_prompt is None:
            sys_prompt = "You are an expert in rhetoric and argument analysis. Provide accurate, objective assessments."
        else:
            sys_prompt = system_prompt
            
        PROMPT_TEMPLATE = """
        Analyze the argumentative complexity in the following passage. Consider whether the argument structure is linear (simple, sequential points) or complex (multifaceted, interconnected, nuanced).

        Passage:
        {passage}

        Please perform the following analysis:
        1. Identify the main argument or claim of the passage
        2. Map the argument structure (linear, branching, complex network, etc.)
        3. Evaluate the complexity of reasoning
        4. Identify any sophisticated argumentative techniques (e.g., qualification, concession, refutation)
        5. Rate the argumentative complexity on a scale from 1-5, where:
        - 1: Simple, linear argument with straightforward points
        - 2: Mostly linear with some branching or qualification
        - 3: Moderately complex with multiple connected points
        - 4: Complex argument structure with sophisticated reasoning
        - 5: Highly complex argument with intricate logical structure and nuanced positions
        
        Format your response as JSON with the following structure:
        {{
        "main_claim": "The central argument of the passage",
        "argument_structure": "Description of structure (linear, branching, etc.)",
        "argument_techniques": ["technique1", "technique2"],
        "argument_complexity_score": 3,
        "explanation": "Brief explanation of your rating"
        }}
        
        Return ONLY valid JSON, with NO additional text before or after.
        """
        
        prompt = PROMPT_TEMPLATE.format(passage=text)
        
        # Add a short delay to avoid rate limiting
        time.sleep(0.1)
        
        response = self.anthropic_client.messages.create(
            model=self.claude_params["model"],
            max_tokens=self.claude_params["max_tokens"],
            temperature=self.claude_params["temperature"],
            system=sys_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        
        result_text = response.content[0].text
        result = extract_json_from_response(result_text)
        
        if result is None:
            raise ValueError("Failed to extract valid JSON from API response.")
        
        main_claim = result.get('main_claim', '')
        argument_structure = result.get('argument_structure', '')
        argument_techniques = result.get('argument_techniques', [])
        complexity_score = result.get('argument_complexity_score', 0)
        explanation = result.get('explanation', '')
        
        result_dict = {
            'passage_id': passage_id,
            'title': title,
            'main_claim': main_claim,
            'argument_structure': argument_structure,
            'argument_techniques': argument_techniques,
            'argument_complexity_score': complexity_score,
            'explanation': explanation
        }
        
        # Update metrics dictionary
        existing_metrics = next((item for item in self.metrics['rhetorical_structure'] if item.get('passage_id') == passage_id), None)
        if existing_metrics:
            existing_metrics['argumentative_complexity'] = complexity_score
        else:
            self.metrics['rhetorical_structure'].append({
                'passage_id': passage_id,
                'title': title,
                'argumentative_complexity': complexity_score
            })
            
        return result_dict
        
    except Exception as e:
        print(f"Exception when calculating argumentative complexity for passage index {passage_index}: {e}")
        return {}

# Add this function to the SATPassageAnalyzer class
SATPassageAnalyzer.calculate_argumentative_complexity_single = calculate_argumentative_complexity_single

def calculate_argumentative_complexity_parallel(self, max_workers: int = 5, system_prompt: Optional[str] = None) -> List[Dict[str, Any]]:
    """Calculate argumentative complexity for all passages in parallel."""
    if not hasattr(self, 'anthropic_client') or not self.anthropic_client:
        print("Error: Anthropic client not initialized. Call init_anthropic_client() first.")
        return []
    
    return self.parallel_process_metric(
        metric_name="argumentative_complexity",
        process_function=self.calculate_argumentative_complexity_single,
        max_workers=max_workers,
        system_prompt=system_prompt
    )

# Add this function to the SATPassageAnalyzer class
SATPassageAnalyzer.calculate_argumentative_complexity_parallel = calculate_argumentative_complexity_parallel

def calculate_organizational_clarity_single(self, passage_index: int, system_prompt: Optional[str] = None) -> Dict[str, Any]:
    """Process a single passage for organizational clarity."""
    try:
        passage = self.passages[passage_index]
        if 'text' not in passage:
            print(f"Warning: No 'text' field found for passage {passage.get('id', 'unknown')}")
            return {}
            
        text = passage['text']
        passage_id = passage.get('id', passage.get('passage_id', 'unknown'))
        title = passage.get('title', f"Passage {passage_id}")
        
        if system_prompt is None:
            sys_prompt = "You are an expert in text structure and composition. Provide accurate, objective assessments."
        else:
            sys_prompt = system_prompt
            
        PROMPT_TEMPLATE = """
        Analyze the organizational clarity in the following passage. Consider how transparent and easy to follow the structure is.

        Passage:
        {passage}

        Please perform the following analysis:
        1. Identify the organizational pattern used (e.g., chronological, compare/contrast, problem/solution)
        2. Evaluate the clarity of the overall structure
        3. Assess how effectively paragraphs are organized
        4. Note the presence of structural markers (topic sentences, headings, etc.)
        5. Rate the organizational clarity on a scale from 1-5, where:
        - 1: Unclear, disorganized structure that's difficult to follow
        - 2: Somewhat unclear structure with some organizational issues
        - 3: Moderately clear organization with a discernible structure
        - 4: Clear, logical organization that's easy to follow
        - 5: Exceptionally clear, sophisticated organization with elegant structure
        
        Note that this scale is reversed from most other scales - a higher score means CLEARER organization, not more complex or difficult.
        
        Format your response as JSON with the following structure:
        {{
        "organizational_pattern": "type of organization used",
        "structural_markers": ["marker1", "marker2"],
        "paragraph_organization": "assessment of paragraph structure",
        "organizational_clarity_score": 4,
        "explanation": "Brief explanation of your rating"
        }}
        
        Return ONLY valid JSON, with NO additional text before or after.
        """
        
        prompt = PROMPT_TEMPLATE.format(passage=text)
        
        # Add a short delay to avoid rate limiting
        time.sleep(0.1)
        
        response = self.anthropic_client.messages.create(
            model=self.claude_params["model"],
            max_tokens=self.claude_params["max_tokens"],
            temperature=self.claude_params["temperature"],
            system=sys_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        
        result_text = response.content[0].text
        result = extract_json_from_response(result_text)
        
        if result is None:
            raise ValueError("Failed to extract valid JSON from API response.")
        
        organizational_pattern = result.get('organizational_pattern', '')
        structural_markers = result.get('structural_markers', [])
        paragraph_organization = result.get('paragraph_organization', '')
        clarity_score = result.get('organizational_clarity_score', 0)
        explanation = result.get('explanation', '')
        
        # For difficulty measure, invert the score (5 becomes 1, 4 becomes 2, etc.)
        # since higher clarity means lower difficulty
        inverted_score = 6 - clarity_score if clarity_score is not None and clarity_score > 0 else None
        
        result_dict = {
            'passage_id': passage_id,
            'title': title,
            'organizational_pattern': organizational_pattern,
            'structural_markers': structural_markers,
            'paragraph_organization': paragraph_organization,
            'organizational_clarity_score': clarity_score,
            'inverted_score_for_difficulty': inverted_score,
            'explanation': explanation
        }
        
        # Update metrics dictionary
        existing_metrics = next((item for item in self.metrics['rhetorical_structure'] if item.get('passage_id') == passage_id), None)
        if existing_metrics:
            existing_metrics['organizational_clarity'] = inverted_score
        else:
            self.metrics['rhetorical_structure'].append({
                'passage_id': passage_id,
                'title': title,
                'organizational_clarity': inverted_score
            })
            
        return result_dict
        
    except Exception as e:
        print(f"Exception when calculating organizational clarity for passage index {passage_index}: {e}")
        return {}

# Add this function to the SATPassageAnalyzer class
SATPassageAnalyzer.calculate_organizational_clarity_single = calculate_organizational_clarity_single

def calculate_organizational_clarity_parallel(self, max_workers: int = 5, system_prompt: Optional[str] = None) -> List[Dict[str, Any]]:
    """Calculate organizational clarity for all passages in parallel."""
    if not hasattr(self, 'anthropic_client') or not self.anthropic_client:
        print("Error: Anthropic client not initialized. Call init_anthropic_client() first.")
        return []
    
    return self.parallel_process_metric(
        metric_name="organizational_clarity",
        process_function=self.calculate_organizational_clarity_single,
        max_workers=max_workers,
        system_prompt=system_prompt
    )

# Add this function to the SATPassageAnalyzer class
SATPassageAnalyzer.calculate_organizational_clarity_parallel = calculate_organizational_clarity_parallel

def calculate_transitional_elements_single(self, passage_index: int, system_prompt: Optional[str] = None) -> Dict[str, Any]:
    """Process a single passage for transitional elements."""
    try:
        passage = self.passages[passage_index]
        if 'text' not in passage:
            print(f"Warning: No 'text' field found for passage {passage.get('id', 'unknown')}")
            return {}
            
        text = passage['text']
        passage_id = passage.get('id', passage.get('passage_id', 'unknown'))
        title = passage.get('title', f"Passage {passage_id}")
        
        if system_prompt is None:
            sys_prompt = "You are an expert in composition and textual analysis. Provide accurate, objective assessments."
        else:
            sys_prompt = system_prompt
            
        PROMPT_TEMPLATE = """
        Analyze the transitional elements in the following passage. Consider the clarity and sophistication of transitions between ideas, paragraphs, and sections.

        Passage:
        {passage}

        Please perform the following analysis:
        1. Identify all transitional words, phrases, and sentences in the passage
        2. Categorize the types of transitions used (e.g., additive, causal, sequential)
        3. Evaluate the sophistication of these transitions
        4. Assess how effectively they connect ideas and paragraphs
        5. Rate the clarity and sophistication of transitions on a scale from 1-5, where:
        - 1: Few or basic transitions, abrupt shifts between ideas
        - 2: Some simple transitions with occasionally unclear connections
        - 3: Adequate transitions with generally clear connections
        - 4: Well-developed transitions creating smooth flow between ideas
        - 5: Sophisticated, varied transitions creating seamless connections throughout
        
        Format your response as JSON with the following structure:
        {{
        "transitional_elements": [
            {{"transition": "example transition", "type": "type of transition", "sophistication": "basic/moderate/sophisticated"}}
        ],
        "transition_count": 8,
        "transition_types": ["additive", "causal", "sequential"],
        "transitional_elements_score": 3,
        "explanation": "Brief explanation of your rating"
        }}
        
        Return ONLY valid JSON, with NO additional text before or after.
        """
        
        prompt = PROMPT_TEMPLATE.format(passage=text)
        
        # Add a short delay to avoid rate limiting
        time.sleep(0.1)
        
        response = self.anthropic_client.messages.create(
            model=self.claude_params["model"],
            max_tokens=self.claude_params["max_tokens"],
            temperature=self.claude_params["temperature"],
            system=sys_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        
        result_text = response.content[0].text
        result = extract_json_from_response(result_text)
        
        if result is None:
            raise ValueError("Failed to extract valid JSON from API response.")
        
        transitional_elements = result.get('transitional_elements', [])
        transition_count = result.get('transition_count', 0)
        transition_types = result.get('transition_types', [])
        transition_score = result.get('transitional_elements_score', 0)
        explanation = result.get('explanation', '')
        
        result_dict = {
            'passage_id': passage_id,
            'title': title,
            'transitional_elements': transitional_elements,
            'transition_count': transition_count,
            'transition_types': transition_types,
            'transitional_elements_score': transition_score,
            'explanation': explanation
        }
        
        # Update metrics dictionary
        existing_metrics = next((item for item in self.metrics['rhetorical_structure'] if item.get('passage_id') == passage_id), None)
        if existing_metrics:
            existing_metrics['transitional_elements'] = transition_score
        else:
            self.metrics['rhetorical_structure'].append({
                'passage_id': passage_id,
                'title': title,
                'transitional_elements': transition_score
            })
            
        return result_dict
        
    except Exception as e:
        print(f"Exception when calculating transitional elements for passage index {passage_index}: {e}")
        return {}

# Add this function to the SATPassageAnalyzer class
SATPassageAnalyzer.calculate_transitional_elements_single = calculate_transitional_elements_single

def calculate_transitional_elements_parallel(self, max_workers: int = 5, system_prompt: Optional[str] = None) -> List[Dict[str, Any]]:
    """Calculate transitional elements for all passages in parallel."""
    if not hasattr(self, 'anthropic_client') or not self.anthropic_client:
        print("Error: Anthropic client not initialized. Call init_anthropic_client() first.")
        return []
    
    return self.parallel_process_metric(
        metric_name="transitional_elements",
        process_function=self.calculate_transitional_elements_single,
        max_workers=max_workers,
        system_prompt=system_prompt
    )

# Add this function to the SATPassageAnalyzer class
SATPassageAnalyzer.calculate_transitional_elements_parallel = calculate_transitional_elements_parallel

def calculate_prior_knowledge_requirements_single(self, passage_index: int, system_prompt: Optional[str] = None) -> Dict[str, Any]:
    """Process a single passage for prior knowledge requirements."""
    try:
        passage = self.passages[passage_index]
        if 'text' not in passage:
            print(f"Warning: No 'text' field found for passage {passage.get('id', 'unknown')}")
            return {}
            
        text = passage['text']
        passage_id = passage.get('id', passage.get('passage_id', 'unknown'))
        title = passage.get('title', f"Passage {passage_id}")
        
        if system_prompt is None:
            sys_prompt = "You are an expert in educational assessment and content analysis. Provide accurate, objective assessments for high school students preparing for the SAT."
        else:
            sys_prompt = system_prompt
            
        PROMPT_TEMPLATE = """
        Analyze the prior knowledge requirements in the following passage. Consider what background knowledge a reader would need to fully understand the content.

        Passage:
        {passage}

        Please perform the following analysis:
        1. Identify specific facts, concepts, or information that a reader would need to know beforehand
        2. Categorize these knowledge requirements by domain (e.g., science, history, literature)
        3. Assess how essential this background knowledge is for comprehension
        4. Rate the prior knowledge requirements on a scale from 1-5, where:
        - 1: Minimal prior knowledge needed, accessible to most readers
        - 2: Some basic prior knowledge needed, but context provides sufficient clues
        - 3: Moderate prior knowledge needed for full comprehension
        - 4: Substantial prior knowledge needed across multiple domains
        - 5: Extensive specialized prior knowledge required for comprehension
        
        Format your response as JSON with the following structure:
        {{
        "required_knowledge": [
            {{"knowledge": "specific fact or concept", "domain": "knowledge domain", "essentiality": "low/medium/high"}}
        ],
        "knowledge_domains": ["domain1", "domain2"],
        "prior_knowledge_score": 3,
        "explanation": "Brief explanation of your rating"
        }}
        
        Return ONLY valid JSON, with NO additional text before or after.
        """
        
        prompt = PROMPT_TEMPLATE.format(passage=text)
        
        # Add a short delay to avoid rate limiting
        time.sleep(0.1)
        
        response = self.anthropic_client.messages.create(
            model=self.claude_params["model"],
            max_tokens=self.claude_params["max_tokens"],
            temperature=self.claude_params["temperature"],
            system=sys_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        
        result_text = response.content[0].text
        result = extract_json_from_response(result_text)
        
        if result is None:
            raise ValueError("Failed to extract valid JSON from API response.")
        
        required_knowledge = result.get('required_knowledge', [])
        knowledge_domains = result.get('knowledge_domains', [])
        prior_knowledge_score = result.get('prior_knowledge_score', 0)
        explanation = result.get('explanation', '')
        
        result_dict = {
            'passage_id': passage_id,
            'title': title,
            'required_knowledge': required_knowledge,
            'knowledge_domains': knowledge_domains,
            'prior_knowledge_score': prior_knowledge_score,
            'explanation': explanation
        }
        
        # Update metrics dictionary
        existing_metrics = next((item for item in self.metrics['content_accessibility'] if item.get('passage_id') == passage_id), None)
        if existing_metrics:
            existing_metrics['prior_knowledge_requirements'] = prior_knowledge_score
        else:
            self.metrics['content_accessibility'].append({
                'passage_id': passage_id,
                'title': title,
                'prior_knowledge_requirements': prior_knowledge_score
            })
            
        return result_dict
        
    except Exception as e:
        print(f"Exception when calculating prior knowledge requirements for passage index {passage_index}: {e}")
        return {}

# Add this function to the SATPassageAnalyzer class
SATPassageAnalyzer.calculate_prior_knowledge_requirements_single = calculate_prior_knowledge_requirements_single

def calculate_prior_knowledge_requirements_parallel(self, max_workers: int = 5, system_prompt: Optional[str] = None) -> List[Dict[str, Any]]:
    """Calculate prior knowledge requirements for all passages in parallel."""
    if not hasattr(self, 'anthropic_client') or not self.anthropic_client:
        print("Error: Anthropic client not initialized. Call init_anthropic_client() first.")
        return []
    
    return self.parallel_process_metric(
        metric_name="prior_knowledge_requirements",
        process_function=self.calculate_prior_knowledge_requirements_single,
        max_workers=max_workers,
        system_prompt=system_prompt
    )

# Add this function to the SATPassageAnalyzer class
SATPassageAnalyzer.calculate_prior_knowledge_requirements_parallel = calculate_prior_knowledge_requirements_parallel

def calculate_disciplinary_perspective_single(self, passage_index: int, system_prompt: Optional[str] = None) -> Dict[str, Any]:
    """Process a single passage for disciplinary perspective."""
    try:
        passage = self.passages[passage_index]
        if 'text' not in passage:
            print(f"Warning: No 'text' field found for passage {passage.get('id', 'unknown')}")
            return {}
            
        text = passage['text']
        passage_id = passage.get('id', passage.get('passage_id', 'unknown'))
        title = passage.get('title', f"Passage {passage_id}")
        
        if system_prompt is None:
            sys_prompt = "You are an expert in academic disciplines and educational assessment. Provide accurate, objective assessments for high school students."
        else:
            sys_prompt = system_prompt
            
        PROMPT_TEMPLATE = """
        Analyze the disciplinary perspective in the following passage. Consider what field-specific thought patterns and approaches are required to fully understand the content.

        Passage:
        {passage}

        Please perform the following analysis:
        1. Identify the primary academic discipline(s) represented in the passage
        2. Evaluate the extent to which the passage uses field-specific terminology, methods, or reasoning
        3. Assess how essential disciplinary knowledge is to understanding the passage
        4. Rate the disciplinary perspective requirements on a scale from 1-5, where:
        - 1: Minimal field-specific knowledge required, accessible across disciplines
        - 2: Some field-specific elements but explained in general terms
        - 3: Moderate field-specific approach with some specialized thinking required
        - 4: Strong disciplinary perspective with substantial specialized thinking
        - 5: Highly specialized disciplinary perspective requiring expert-level understanding
        
        Format your response as JSON with the following structure:
        {{
        "primary_disciplines": ["discipline1", "discipline2"],
        "field_specific_elements": [
            {{"element": "specific term or concept", "discipline": "relevant discipline", "explanation": "brief explanation"}}
        ],
        "disciplinary_perspective_score": 3,
        "explanation": "Brief explanation of your rating"
        }}
        
        Return ONLY valid JSON, with NO additional text before or after.
        """
        
        prompt = PROMPT_TEMPLATE.format(passage=text)
        
        # Add a short delay to avoid rate limiting
        time.sleep(0.1)
        
        response = self.anthropic_client.messages.create(
            model=self.claude_params["model"],
            max_tokens=self.claude_params["max_tokens"],
            temperature=self.claude_params["temperature"],
            system=sys_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        
        result_text = response.content[0].text
        result = extract_json_from_response(result_text)
        
        if result is None:
            raise ValueError("Failed to extract valid JSON from API response.")
        
        primary_disciplines = result.get('primary_disciplines', [])
        field_specific_elements = result.get('field_specific_elements', [])
        disciplinary_score = result.get('disciplinary_perspective_score', 0)
        explanation = result.get('explanation', '')
        
        result_dict = {
            'passage_id': passage_id,
            'title': title,
            'primary_disciplines': primary_disciplines,
            'field_specific_elements': field_specific_elements,
            'disciplinary_perspective_score': disciplinary_score,
            'explanation': explanation
        }
        
        # Update metrics dictionary
        existing_metrics = next((item for item in self.metrics['content_accessibility'] if item.get('passage_id') == passage_id), None)
        if existing_metrics:
            existing_metrics['disciplinary_perspective'] = disciplinary_score
        else:
            self.metrics['content_accessibility'].append({
                'passage_id': passage_id,
                'title': title,
                'disciplinary_perspective': disciplinary_score
            })
            
        return result_dict
        
    except Exception as e:
        print(f"Exception when calculating disciplinary perspective for passage index {passage_index}: {e}")
        return {}

# Add this function to the SATPassageAnalyzer class
SATPassageAnalyzer.calculate_disciplinary_perspective_single = calculate_disciplinary_perspective_single

def calculate_disciplinary_perspective_parallel(self, max_workers: int = 5, system_prompt: Optional[str] = None) -> List[Dict[str, Any]]:
    """Calculate disciplinary perspective for all passages in parallel."""
    if not hasattr(self, 'anthropic_client') or not self.anthropic_client:
        print("Error: Anthropic client not initialized. Call init_anthropic_client() first.")
        return []
    
    return self.parallel_process_metric(
        metric_name="disciplinary_perspective",
        process_function=self.calculate_disciplinary_perspective_single,
        max_workers=max_workers,
        system_prompt=system_prompt
    )

# Add this function to the SATPassageAnalyzer class
SATPassageAnalyzer.calculate_disciplinary_perspective_parallel = calculate_disciplinary_perspective_parallel

def calculate_language_modernity_single(self, passage_index: int, system_prompt: Optional[str] = None) -> Dict[str, Any]:
    """Process a single passage for language modernity."""
    try:
        passage = self.passages[passage_index]
        if 'text' not in passage:
            print(f"Warning: No 'text' field found for passage {passage.get('id', 'unknown')}")
            return {}
            
        text = passage['text']
        passage_id = passage.get('id', passage.get('passage_id', 'unknown'))
        title = passage.get('title', f"Passage {passage_id}")
        
        if system_prompt is None:
            sys_prompt = "You are an expert in linguistic analysis and historical language development. Provide accurate, objective assessments for high school students."
        else:
            sys_prompt = system_prompt
            
        PROMPT_TEMPLATE = """
        Analyze the language modernity in the following passage. Consider the spectrum from contemporary to archaic language use.

        Passage:
        {passage}

        Please perform the following analysis:
        1. Identify archaic or dated language elements (vocabulary, grammar, syntax, idioms)
        2. Determine the approximate time period the language represents
        3. Evaluate how accessible the language would be to modern high school students
        4. Rate the language modernity on a scale from 1-5, where:
        - 1: Fully contemporary language, current vocabulary and expressions
        - 2: Mostly contemporary with occasional older usages
        - 3: Mix of contemporary and somewhat dated language
        - 4: Predominantly older or somewhat archaic language
        - 5: Highly archaic language that would be challenging for modern readers
        
        Format your response as JSON with the following structure:
        {{
        "archaic_elements": [
            {{"element": "specific word or phrase", "contemporary_equivalent": "modern version"}}
        ],
        "approximate_period": "time period the language represents",
        "language_modernity_score": 3,
        "explanation": "Brief explanation of your rating"
        }}
        
        Return ONLY valid JSON, with NO additional text before or after.
        """
        
        prompt = PROMPT_TEMPLATE.format(passage=text)
        
        # Add a short delay to avoid rate limiting
        time.sleep(0.1)
        
        response = self.anthropic_client.messages.create(
            model=self.claude_params["model"],
            max_tokens=self.claude_params["max_tokens"],
            temperature=self.claude_params["temperature"],
            system=sys_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        
        result_text = response.content[0].text
        result = extract_json_from_response(result_text)
        
        if result is None:
            raise ValueError("Failed to extract valid JSON from API response.")
        
        archaic_elements = result.get('archaic_elements', [])
        approximate_period = result.get('approximate_period', '')
        modernity_score = result.get('language_modernity_score', 0)
        explanation = result.get('explanation', '')
        
        result_dict = {
            'passage_id': passage_id,
            'title': title,
            'archaic_elements': archaic_elements,
            'approximate_period': approximate_period,
            'language_modernity_score': modernity_score,
            'explanation': explanation
        }
        
        # Update metrics dictionary
        existing_metrics = next((item for item in self.metrics['content_accessibility'] if item.get('passage_id') == passage_id), None)
        if existing_metrics:
            existing_metrics['language_modernity'] = modernity_score
        else:
            self.metrics['content_accessibility'].append({
                'passage_id': passage_id,
                'title': title,
                'language_modernity': modernity_score
            })
            
        return result_dict
        
    except Exception as e:
        print(f"Exception when calculating language modernity for passage index {passage_index}: {e}")
        return {}

# Add this function to the SATPassageAnalyzer class
SATPassageAnalyzer.calculate_language_modernity_single = calculate_language_modernity_single

def calculate_language_modernity_parallel(self, max_workers: int = 5, system_prompt: Optional[str] = None) -> List[Dict[str, Any]]:
    """Calculate language modernity for all passages in parallel."""
    if not hasattr(self, 'anthropic_client') or not self.anthropic_client:
        print("Error: Anthropic client not initialized. Call init_anthropic_client() first.")
        return []
    
    return self.parallel_process_metric(
        metric_name="language_modernity",
        process_function=self.calculate_language_modernity_single,
        max_workers=max_workers,
        system_prompt=system_prompt
    )

# Add this function to the SATPassageAnalyzer class
SATPassageAnalyzer.calculate_language_modernity_parallel = calculate_language_modernity_parallel

def calculate_inference_requirement_single(self, passage_index: int, system_prompt: Optional[str] = None) -> Dict[str, Any]:
    """Process a single passage for inference requirement."""
    try:
        passage = self.passages[passage_index]
        if 'text' not in passage:
            print(f"Warning: No 'text' field found for passage {passage.get('id', 'unknown')}")
            return {}
            
        text = passage['text']
        passage_id = passage.get('id', passage.get('passage_id', 'unknown'))
        title = passage.get('title', f"Passage {passage_id}")
        
        if system_prompt is None:
            sys_prompt = "You are an expert in reading comprehension and cognitive analysis. Provide accurate, objective assessments for high school students taking the SAT."
        else:
            sys_prompt = system_prompt
            
        PROMPT_TEMPLATE = """
        Analyze the inference requirements in the following passage. Consider how much reading between the lines is needed to fully understand the content.

        Passage:
        {passage}

        Please perform the following analysis:
        1. Identify information that must be inferred rather than being explicitly stated
        2. Determine the types of inferences required (e.g., causal, predictive, character-based)
        3. Evaluate the cognitive demand of making these inferences
        4. Rate the inference requirement on a scale from 1-5, where:
        - 1: Minimal inference needed, mostly explicit information
        - 2: Some basic inference required but main points are explicit
        - 3: Moderate inference needed for full comprehension
        - 4: Substantial inference required throughout the passage
        - 5: Heavy reliance on inference with little explicit guidance
        
        Format your response as JSON with the following structure:
        {{
        "inference_examples": [
            {{"inference": "example of required inference", "type": "type of inference", "difficulty": "basic/moderate/complex"}}
        ],
        "inference_types": ["causal", "character-based"],
        "inference_requirement_score": 3,
        "explanation": "Brief explanation of your rating"
        }}
        
        Return ONLY valid JSON, with NO additional text before or after.
        """
        
        prompt = PROMPT_TEMPLATE.format(passage=text)
        
        # Add a short delay to avoid rate limiting
        time.sleep(0.1)
        
        response = self.anthropic_client.messages.create(
            model=self.claude_params["model"],
            max_tokens=self.claude_params["max_tokens"],
            temperature=self.claude_params["temperature"],
            system=sys_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        
        result_text = response.content[0].text
        result = extract_json_from_response(result_text)
        
        if result is None:
            raise ValueError("Failed to extract valid JSON from API response.")
        
        inference_examples = result.get('inference_examples', [])
        inference_types = result.get('inference_types', [])
        inference_score = result.get('inference_requirement_score', 0)
        explanation = result.get('explanation', '')
        
        result_dict = {
            'passage_id': passage_id,
            'title': title,
            'inference_examples': inference_examples,
            'inference_types': inference_types,
            'inference_requirement_score': inference_score,
            'explanation': explanation
        }
        
        # Update metrics dictionary
        existing_metrics = next((item for item in self.metrics['cognitive_demands'] if item.get('passage_id') == passage_id), None)
        if existing_metrics:
            existing_metrics['inference_requirement'] = inference_score
        else:
            self.metrics['cognitive_demands'].append({
                'passage_id': passage_id,
                'title': title,
                'inference_requirement': inference_score
            })
            
        return result_dict
        
    except Exception as e:
        print(f"Exception when calculating inference requirement for passage index {passage_index}: {e}")
        return {}

# Add this function to the SATPassageAnalyzer class
SATPassageAnalyzer.calculate_inference_requirement_single = calculate_inference_requirement_single

def calculate_inference_requirement_parallel(self, max_workers: int = 5, system_prompt: Optional[str] = None) -> List[Dict[str, Any]]:
    """Calculate inference requirement for all passages in parallel."""
    if not hasattr(self, 'anthropic_client') or not self.anthropic_client:
        print("Error: Anthropic client not initialized. Call init_anthropic_client() first.")
        return []
    
    return self.parallel_process_metric(
        metric_name="inference_requirement",
        process_function=self.calculate_inference_requirement_single,
        max_workers=max_workers,
        system_prompt=system_prompt
    )

# Add this function to the SATPassageAnalyzer class
SATPassageAnalyzer.calculate_inference_requirement_parallel = calculate_inference_requirement_parallel

def calculate_figurative_language_single(self, passage_index: int, system_prompt: Optional[str] = None) -> Dict[str, Any]:
    """Process a single passage for figurative language."""
    try:
        passage = self.passages[passage_index]
        if 'text' not in passage:
            print(f"Warning: No 'text' field found for passage {passage.get('id', 'unknown')}")
            return {}
            
        text = passage['text']
        passage_id = passage.get('id', passage.get('passage_id', 'unknown'))
        title = passage.get('title', f"Passage {passage_id}")
        
        if system_prompt is None:
            sys_prompt = "You are an expert in literary analysis and rhetoric. Provide accurate, objective assessments for high school students."
        else:
            sys_prompt = system_prompt
            
        PROMPT_TEMPLATE = """
        Analyze the figurative language in the following passage. Consider the density and types of metaphors, similes, personification, and other figurative devices.

        Passage:
        {passage}

        Please perform the following analysis:
        1. Identify all instances of figurative language in the passage
        2. Categorize the types of figurative devices used
        3. Evaluate the complexity of the figurative language
        4. Rate the figurative language density on a scale from 1-5, where:
        - 1: Minimal figurative language, mostly literal
        - 2: Some simple figurative devices
        - 3: Moderate use of varied figurative language
        - 4: Frequent use of complex figurative language
        - 5: Dense, sophisticated figurative language throughout
        
        Format your response as JSON with the following structure:
        {{
        "figurative_language": [
            {{"example": "text with figurative language", "type": "metaphor/simile/etc.", "complexity": "simple/moderate/complex"}}
        ],
        "device_types": ["metaphor", "simile", "personification"],
        "device_count": 7,
        "figurative_language_score": 3,
        "explanation": "Brief explanation of your rating"
        }}
        
        Return ONLY valid JSON, with NO additional text before or after.
        """
        
        prompt = PROMPT_TEMPLATE.format(passage=text)
        
        # Add a short delay to avoid rate limiting
        time.sleep(0.1)
        
        response = self.anthropic_client.messages.create(
            model=self.claude_params["model"],
            max_tokens=self.claude_params["max_tokens"],
            temperature=self.claude_params["temperature"],
            system=sys_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        
        result_text = response.content[0].text
        result = extract_json_from_response(result_text)
        
        if result is None:
            raise ValueError("Failed to extract valid JSON from API response.")
        
        figurative_language = result.get('figurative_language', [])
        device_types = result.get('device_types', [])
        device_count = result.get('device_count', 0)
        figurative_score = result.get('figurative_language_score', 0)
        explanation = result.get('explanation', '')
        
        result_dict = {
            'passage_id': passage_id,
            'title': title,
            'figurative_language': figurative_language,
            'device_types': device_types,
            'device_count': device_count,
            'figurative_language_score': figurative_score,
            'explanation': explanation
        }
        
        # Update metrics dictionary
        existing_metrics = next((item for item in self.metrics['cognitive_demands'] if item.get('passage_id') == passage_id), None)
        if existing_metrics:
            existing_metrics['figurative_language'] = figurative_score
        else:
            self.metrics['cognitive_demands'].append({
                'passage_id': passage_id,
                'title': title,
                'figurative_language': figurative_score
            })
            
        return result_dict
        
    except Exception as e:
        print(f"Exception when calculating figurative language for passage index {passage_index}: {e}")
        return {}

# Add this function to the SATPassageAnalyzer class
SATPassageAnalyzer.calculate_figurative_language_single = calculate_figurative_language_single

def calculate_figurative_language_parallel(self, max_workers: int = 5, system_prompt: Optional[str] = None) -> List[Dict[str, Any]]:
    """Calculate figurative language for all passages in parallel."""
    if not hasattr(self, 'anthropic_client') or not self.anthropic_client:
        print("Error: Anthropic client not initialized. Call init_anthropic_client() first.")
        return []
    
    return self.parallel_process_metric(
        metric_name="figurative_language",
        process_function=self.calculate_figurative_language_single,
        max_workers=max_workers,
        system_prompt=system_prompt
    )

# Add this function to the SATPassageAnalyzer class
SATPassageAnalyzer.calculate_figurative_language_parallel = calculate_figurative_language_parallel

def calculate_authors_purpose_single(self, passage_index: int, system_prompt: Optional[str] = None) -> Dict[str, Any]:
    """Process a single passage for author's purpose."""
    try:
        passage = self.passages[passage_index]
        if 'text' not in passage:
            print(f"Warning: No 'text' field found for passage {passage.get('id', 'unknown')}")
            return {}
            
        text = passage['text']
        passage_id = passage.get('id', passage.get('passage_id', 'unknown'))
        title = passage.get('title', f"Passage {passage_id}")
        
        if system_prompt is None:
            sys_prompt = "You are an expert in rhetorical analysis and textual interpretation. Provide accurate, objective assessments for high school students."
        else:
            sys_prompt = system_prompt
            
        PROMPT_TEMPLATE = """
        Analyze the clarity of the author's purpose in the following passage. Consider how obvious versus obscure the author's intent is.

        Passage:
        {passage}

        Please perform the following analysis:
        1. Identify the author's apparent purpose(s) or intent
        2. Evaluate how clearly this purpose is communicated
        3. Assess how much interpretation is required to discern the author's intent
        4. Rate the clarity vs. obscurity of author's purpose on a scale from 1-5, where:
        - 1: Completely transparent purpose, explicitly stated
        - 2: Mostly clear purpose with minimal interpretation needed
        - 3: Moderately clear purpose requiring some interpretation
        - 4: Somewhat obscure purpose requiring significant interpretation
        - 5: Highly obscure or ambiguous purpose requiring extensive interpretation
        
        Format your response as JSON with the following structure:
        {{
        "apparent_purposes": ["primary purpose", "secondary purpose"],
        "purpose_indicators": ["example text indicating purpose", "another indicator"],
        "authors_purpose_score": 3,
        "explanation": "Brief explanation of your rating"
        }}
        
        Return ONLY valid JSON, with NO additional text before or after.
        """
        
        prompt = PROMPT_TEMPLATE.format(passage=text)
        
        # Add a short delay to avoid rate limiting
        time.sleep(0.1)
        
        response = self.anthropic_client.messages.create(
            model=self.claude_params["model"],
            max_tokens=self.claude_params["max_tokens"],
            temperature=self.claude_params["temperature"],
            system=sys_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        
        result_text = response.content[0].text
        result = extract_json_from_response(result_text)
        
        if result is None:
            raise ValueError("Failed to extract valid JSON from API response.")
        
        apparent_purposes = result.get('apparent_purposes', [])
        purpose_indicators = result.get('purpose_indicators', [])
        purpose_score = result.get('authors_purpose_score', 0)
        explanation = result.get('explanation', '')
        
        result_dict = {
            'passage_id': passage_id,
            'title': title,
            'apparent_purposes': apparent_purposes,
            'purpose_indicators': purpose_indicators,
            'authors_purpose_score': purpose_score,
            'explanation': explanation
        }
        
        # Update metrics dictionary
        existing_metrics = next((item for item in self.metrics['cognitive_demands'] if item.get('passage_id') == passage_id), None)
        if existing_metrics:
            existing_metrics['authors_purpose'] = purpose_score
        else:
            self.metrics['cognitive_demands'].append({
                'passage_id': passage_id,
                'title': title,
                'authors_purpose': purpose_score
            })
            
        return result_dict
        
    except Exception as e:
        print(f"Exception when calculating author's purpose for passage index {passage_index}: {e}")
        return {}

# Add this function to the SATPassageAnalyzer class
SATPassageAnalyzer.calculate_authors_purpose_single = calculate_authors_purpose_single

def calculate_authors_purpose_parallel(self, max_workers: int = 5, system_prompt: Optional[str] = None) -> List[Dict[str, Any]]:
    """Calculate author's purpose for all passages in parallel."""
    if not hasattr(self, 'anthropic_client') or not self.anthropic_client:
        print("Error: Anthropic client not initialized. Call init_anthropic_client() first.")
        return []
    
    return self.parallel_process_metric(
        metric_name="authors_purpose",
        process_function=self.calculate_authors_purpose_single,
        max_workers=max_workers,
        system_prompt=system_prompt
    )

# Add this function to the SATPassageAnalyzer class
SATPassageAnalyzer.calculate_authors_purpose_parallel = calculate_authors_purpose_parallel

def calculate_lexile_scores_single(self, passage_index: int) -> Dict[str, Any]:
    """Process a single passage for Lexile score."""
    try:
        passage = self.passages[passage_index]
        if 'text' not in passage:
            print(f"Warning: No 'text' field found for passage {passage.get('id', 'unknown')}")
            return {}
            
        text = passage['text']
        passage_id = passage.get('id', passage.get('passage_id', 'unknown'))
        title = passage.get('title', f"Passage {passage_id}")
        
        # Using the Eigen API for Lexile scores
        api_url = "https://composer.api.eigen.net/api/utils/readability"
        
        # Add a short delay to avoid rate limiting
        time.sleep(0.1)
        
        import requests
        response = requests.post(api_url, json={"text": text}, headers={"Content-Type": "application/json"})
        
        if response.status_code == 200:
            data = response.json()
            lexile_score = data.get('lexile', None)
            
            # ADDED GUARDRAIL: Cap extremely high Lexile scores at 1501
            if lexile_score is not None and lexile_score > 2000:
                print(f"Extremely high Lexile score detected for passage {passage_id}: {lexile_score}, capping at 1501")
                lexile_score = 1501
                
            score_data = {
                'passage_id': passage_id,
                'title': title,
                'lexile_score': lexile_score
            }
            
            # Update metrics dictionary
            existing_metrics = next((item for item in self.metrics['readability'] if item.get('passage_id') == passage_id), None)
            if existing_metrics:
                existing_metrics['lexile_score'] = lexile_score
            else:
                self.metrics['readability'].append(score_data)
                
            return score_data
        else:
            print(f"API Error for passage {passage_id}: {response.status_code} - {response.text}")
            return {
                'passage_id': passage_id,
                'title': title,
                'lexile_score': None,
                'error': f"API Error: {response.status_code}"
            }
            
    except Exception as e:
        print(f"Exception when calculating Lexile score for passage index {passage_index}: {e}")
        passage_id = self.passages[passage_index].get('id', self.passages[passage_index].get('passage_id', 'unknown'))
        title = self.passages[passage_index].get('title', f"Passage {passage_id}")
        return {
            'passage_id': passage_id,
            'title': title,
            'lexile_score': None,
            'error': str(e)
        }

# Add this function to the SATPassageAnalyzer class
SATPassageAnalyzer.calculate_lexile_scores_single = calculate_lexile_scores_single

def calculate_lexile_scores_parallel(self, max_workers: int = 5) -> List[Dict[str, Any]]:
    """Calculate Lexile scores for all passages in parallel using the Eigen API."""
    print(f"\n=== Calculating Lexile Scores in Parallel (Max Workers: {max_workers}) ===")
    
    lexile_scores = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a list of future objects
        futures = [executor.submit(self.calculate_lexile_scores_single, passage_index=idx) 
                  for idx in range(len(self.passages))]
        
        # Process results as they complete
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                result = future.result()
                if result:
                    lexile_scores.append(result)
                    print(f"Completed Lexile score for passage {i+1}/{len(futures)}")
            except Exception as e:
                print(f"Error processing Lexile score: {e}")
    
    self._save_metric("lexile_scores", lexile_scores)
    return lexile_scores

# Add this function to the SATPassageAnalyzer class
SATPassageAnalyzer.calculate_lexile_scores_parallel = calculate_lexile_scores_parallel

def identify_missing_metrics(self, metric_category: str, metric_name: str) -> List[int]:
    """
    Identify passage indices that are missing a specific metric.
    
    Parameters:
    - metric_category: Category of the metric in self.metrics (e.g., 'readability')
    - metric_name: Name of the specific metric (e.g., 'lexile_score')
    
    Returns:
    - List of passage indices that are missing the metric
    """
    missing_indices = []
    
    for idx, passage in enumerate(self.passages):
        passage_id = passage.get('id', passage.get('passage_id', 'unknown'))
        
        # Check if metric exists for this passage
        metrics_list = self.metrics.get(metric_category, [])
        metric_entry = next((item for item in metrics_list if item.get('passage_id') == passage_id), None)
        
        if metric_entry is None or metric_name not in metric_entry or metric_entry[metric_name] is None:
            missing_indices.append(idx)
    
    return missing_indices

# Add this function to the SATPassageAnalyzer class
SATPassageAnalyzer.identify_missing_metrics = identify_missing_metrics

def run_with_retry(self, metric_function, metric_category: str, metric_name: str, 
                   max_workers: int = 3, max_retries: int = 3, 
                   retry_delay: int = 10, **kwargs) -> List[Dict[str, Any]]:
    """
    Run a metric calculation with automatic retries for missing passages.
    
    Parameters:
    - metric_function: The parallelized function to run (e.g., calculate_lexile_scores_parallel)
    - metric_category: Category of the metric in self.metrics (e.g., 'readability')
    - metric_name: Name of the specific metric (e.g., 'lexile_score')
    - max_workers: Maximum number of parallel workers
    - max_retries: Maximum number of retry attempts
    - retry_delay: Delay between retries in seconds
    - **kwargs: Additional arguments to pass to the metric function
    
    Returns:
    - Combined results from all runs
    """
    results = []
    attempts = 0
    
    while attempts < max_retries:
        # Identify passages missing this metric
        missing_indices = self.identify_missing_metrics(metric_category, metric_name)
        
        if not missing_indices:
            print(f"✅ All passages have {metric_name} calculated!")
            break
        
        attempts += 1
        passage_count = len(self.passages)
        missing_count = len(missing_indices)
        success_rate = ((passage_count - missing_count) / passage_count) * 100 if passage_count > 0 else 0
        
        print(f"\n=== Retry #{attempts} for {metric_name} ===")
        print(f"Missing {missing_count}/{passage_count} passages ({success_rate:.1f}% complete)")
        
        if attempts > 1:
            print(f"Waiting {retry_delay} seconds before retry...")
            time.sleep(retry_delay)
        
        # Only process missing passages
        if metric_name == 'lexile_score':
            # Special case for Lexile scores which has its own implementation
            lexile_results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(self.calculate_lexile_scores_single, passage_index=idx) 
                          for idx in missing_indices]
                
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    try:
                        result = future.result()
                        if result:
                            lexile_results.append(result)
                            print(f"Completed Lexile score for missing passage {i+1}/{len(futures)}")
                    except Exception as e:
                        print(f"Error processing Lexile score: {e}")
            
            if lexile_results:
                self._save_metric("lexile_scores", lexile_results)
                results.extend(lexile_results)
        
        else:
            # For metrics that use the standard pattern
            filtered_results = self.parallel_process_metric(
                metric_name=metric_name,
                process_function=partial(getattr(self, f"calculate_{metric_name}_single"), 
                                         **kwargs),
                max_workers=max_workers,
                missing_indices=missing_indices  # This requires modification to parallel_process_metric
            )
            
            results.extend(filtered_results)
        
        # If we've processed all passages, exit early
        if not self.identify_missing_metrics(metric_category, metric_name):
            print(f"✅ All passages now have {metric_name} calculated!")
            break
    
    if attempts == max_retries and self.identify_missing_metrics(metric_category, metric_name):
        missing_count = len(self.identify_missing_metrics(metric_category, metric_name))
        print(f"⚠️ Warning: After {max_retries} attempts, still missing {missing_count} passages for {metric_name}")
    
    return results

# Add this function to the SATPassageAnalyzer class
SATPassageAnalyzer.run_with_retry = run_with_retry