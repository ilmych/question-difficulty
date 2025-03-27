#!/usr/bin/env python3
import pandas as pd
import numpy as np
import re
import json
import requests
from collections import Counter
from io import StringIO
import logging
logger = logging.getLogger("sat_passage_analyzer")


def extract_json_from_response(response_text):
    """
    Attempts to extract valid JSON from a raw response string using regex.
    
    Parameters:
    - response_text: The raw text from the API response.
    
    Returns:
    - A Python object parsed from the JSON if found, or None otherwise.
    """
    pattern = re.compile(r'(\{.*\})', re.DOTALL)
    match = pattern.search(response_text)
    if match:
        json_text = match.group(1)
        try:
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            logger.error("JSON decoding failed after extraction: {e}", exc_info=True)
            return None
    else:
        logger.info("No JSON object found in response.")
        return None

class SATPassageAnalyzer:
    """
    A framework for analyzing and comparing the difficulty of SAT reading passages
    based on quantitative and qualitative metrics.
    """
    
    def __init__(self):
        """Initialize the SAT passage analyzer."""
        self.passages = []
        self.metrics = {
            "readability": [],
            "vocabulary": [],
            "syntactic_complexity": [],
            "conceptual_density": [],
            "rhetorical_structure": [],
            "content_accessibility": [],
            "cognitive_demands": []
        }
        self.results = {}
    
    def load_data(self, data, file_type='csv'):
        """
        Load passage data from either CSV or JSON format.
        
        Parameters:
        - data: Either a file path or string content of the data
        - file_type: 'csv' or 'json'
        
        Returns:
        - List of passage dictionaries
        """
        if file_type == 'csv':
            return self._parse_csv(data)
        elif file_type == 'json':
            return self._parse_json(data)
        else:
            raise ValueError("Unsupported file type. Please use 'csv' or 'json'.")
    
    def _parse_csv(self, csv_data):
        """Parse CSV data into passage dictionaries."""
        try:
            # Check if csv_data is a file path or a data string
            if csv_data.strip().startswith('id,') or csv_data.strip().startswith('passage_id,'):
                # It's a data string; use StringIO from the io module
                df = pd.read_csv(StringIO(csv_data))
            else:
                # It's a file path
                df = pd.read_csv(csv_data)
            
            # Convert DataFrame to list of dictionaries
            self.passages = df.to_dict('records')
            return self.passages
        
        except Exception as e:
            logger.error(f"Error parsing CSV data: {e}", exc_info=True)
            return []
    
    def _parse_json(self, json_data):
        """Parse JSON data into passage dictionaries."""
        try:
            # Check if json_data is a file path or a JSON string
            if json_data.strip().startswith('[') or json_data.strip().startswith('{'):
                # It's a JSON string
                self.passages = json.loads(json_data)
            else:
                # It's a file path - open with UTF-8 encoding to avoid decoding issues
                with open(json_data, 'r', encoding='utf-8') as f:
                    self.passages = json.load(f)
            # Normalize keys: if a passage uses 'passage' instead of 'text', rename it
            for p in self.passages:
                if 'text' not in p and 'passage' in p:
                    p['text'] = p.pop('passage')
            return self.passages
        except Exception as e:
            logger.error(f"Error parsing JSON data: {e}", exc_info=True)
            return []
    
    def init_anthropic_client(self, api_key=None, model="claude-3-7-sonnet-20250219"):
        """
        Initialize the Anthropic API client for Claude.
        
        Parameters:
        - api_key: API key for Claude (if not provided, will look for ANTHROPIC_API_KEY env variable)
        - model: Claude model to use (default: claude-3-7-sonnet-20250219)
        
        Returns:
        - Anthropic client instance or None if initialization failed
        """
        try:
            import anthropic
            from os import environ
        except ImportError:
            logger.error("Error: anthropic package not installed. Please install with 'pip install anthropic'")
            return None
        
        # Get API key from parameter or environment variable
        api_key = api_key or environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            logger.error("Error: No API key provided. Set ANTHROPIC_API_KEY environment variable or pass api_key parameter")
            return None
        
        try:
            # Initialize the client
            self.anthropic_client = anthropic.Anthropic(api_key=api_key)
            
            # Store default parameters for API calls
            self.claude_params = {
                "model": model,
                "max_tokens": 2000,
                "temperature": 0
            }
            
            logger.info(f"Anthropic API client initialized successfully with model {model}")
            return self.anthropic_client
        except Exception as e:
            logger.error(f"Error initializing Anthropic client: {e}", exc_info=True)
            return None
            
    # ==================== UTILITY FUNCTIONS ====================
    
    def _count_sentences(self, text):
        """Count the number of sentences in text."""
        sentences = re.split(r'[.!?]+\s', text)
        sentences = [s for s in sentences if s.strip()]
        return len(sentences)
    
    def _count_words(self, text):
        """Count the number of words in text."""
        words = re.findall(r'\b[a-z\d\'-]+\b', text.lower())
        return len(words)
    
    def _count_syllables(self, text):
        """Count the total syllables in text."""
        words = re.findall(r'\b[a-z\d\'-]+\b', text.lower())
        syllable_count = sum(self._count_word_syllables(word) for word in words)
        return syllable_count
    
    def _count_word_syllables(self, word):
        """Count syllables in a word using a basic algorithm."""
        word = word.lower()
        if len(word) <= 3:
            return 1
        word = re.sub(r'(?:[^laeiouy]es|ed|[^laeiouy]e)$', '', word)
        word = re.sub(r'y', 'i', word)
        syllables = len(re.findall(r'[aeiouy]+', word))
        if syllables == 0:
            syllables = 1
        return syllables
        
    def calculate_flesch_kincaid(self):
        """Calculate Flesch-Kincaid Grade Level for all passages."""
        fk_scores = []
        for passage in self.passages:
            if 'text' not in passage:
                logger.warning(f"Warning: No 'text' field found for passage {passage.get('id', 'unknown')}")
                continue
            text = passage['text']
            passage_id = passage.get('id', passage.get('passage_id', 'unknown'))
            try:
                word_count = self._count_words(text)
                sentence_count = self._count_sentences(text)
                syllable_count = self._count_syllables(text)
                if sentence_count == 0 or word_count == 0:
                    fk_grade_level = None
                else:
                    fk_grade_level = 0.39 * (word_count / sentence_count) + 11.8 * (syllable_count / word_count) - 15.59
                score_data = {
                    'passage_id': passage_id,
                    'title': passage.get('title', f"Passage {passage_id}"),
                    'flesch_kincaid_grade': fk_grade_level
                }
                fk_scores.append(score_data)
                existing_metrics = next((item for item in self.metrics['readability'] if item.get('passage_id') == passage_id), None)
                if existing_metrics:
                    existing_metrics['flesch_kincaid_grade'] = fk_grade_level
                else:
                    self.metrics['readability'].append(score_data)
            except Exception as e:
                logger.error(f"Exception when calculating Flesch-Kincaid grade for passage {passage_id}: {e}", exc_info=True)
                fk_scores.append({
                    'passage_id': passage_id,
                    'title': passage.get('title', f"Passage {passage_id}"),
                    'flesch_kincaid_grade': None,
                    'error': str(e)
                })
        return fk_scores
        
    def calculate_avg_sentence_length(self):
        """Calculate average sentence length for all passages."""
        sentence_lengths = []
        for passage in self.passages:
            if 'text' not in passage:
                logger.warning(f"Warning: No 'text' field found for passage {passage.get('id', 'unknown')}")
                continue
            text = passage['text']
            passage_id = passage.get('id', passage.get('passage_id', 'unknown'))
            try:
                word_count = self._count_words(text)
                sentence_count = self._count_sentences(text)
                avg_length = None if sentence_count == 0 else word_count / sentence_count
                score_data = {
                    'passage_id': passage_id,
                    'title': passage.get('title', f"Passage {passage_id}"),
                    'avg_sentence_length': avg_length
                }
                sentence_lengths.append(score_data)
                existing_metrics = next((item for item in self.metrics['readability'] if item.get('passage_id') == passage_id), None)
                if existing_metrics:
                    existing_metrics['avg_sentence_length'] = avg_length
                else:
                    self.metrics['readability'].append(score_data)
            except Exception as e:
                logger.error(f"Exception when calculating average sentence length for passage {passage_id}: {e}", exc_info=True)
                sentence_lengths.append({
                    'passage_id': passage_id,
                    'title': passage.get('title', f"Passage {passage_id}"),
                    'avg_sentence_length': None,
                    'error': str(e)
                })
        return sentence_lengths
        
    def calculate_readability_score(self):
        """Calculate a combined readability score based on all readability metrics."""
        readability_scores = []
        if not self.metrics['readability']:
            logger.warning("Warning: No readability metrics calculated yet. Please calculate metrics first.")
            return []
        for passage_metrics in self.metrics['readability']:
            passage_id = passage_metrics.get('passage_id')
            lexile_score = passage_metrics.get('lexile_score')
            fk_grade = passage_metrics.get('flesch_kincaid_grade')
            avg_sentence_length = passage_metrics.get('avg_sentence_length')
            normalized_scores = []
            if lexile_score is not None:
                if lexile_score < 900:
                    normalized_lexile = 1
                elif lexile_score < 1200:
                    normalized_lexile = 2
                elif lexile_score < 1300:
                    normalized_lexile = 3
                elif lexile_score < 1400:
                    normalized_lexile = 4
                else:
                    normalized_lexile = 5
                normalized_scores.append(normalized_lexile)
            if fk_grade is not None:
                if fk_grade < 9:
                    normalized_fk = 1
                elif fk_grade < 11:
                    normalized_fk = 2
                elif fk_grade < 13:
                    normalized_fk = 3
                elif fk_grade < 15:
                    normalized_fk = 4
                else:
                    normalized_fk = 5
                normalized_scores.append(normalized_fk)
            if avg_sentence_length is not None:
                if avg_sentence_length < 18:
                    normalized_asl = 1
                elif avg_sentence_length < 22:
                    normalized_asl = 2
                elif avg_sentence_length < 25:
                    normalized_asl = 3
                elif avg_sentence_length < 30:
                    normalized_asl = 4
                else:
                    normalized_asl = 5
                normalized_scores.append(normalized_asl)
            overall_readability = sum(normalized_scores) / len(normalized_scores) if normalized_scores else None
            passage_metrics['overall_readability'] = overall_readability
            readability_scores.append({
                'passage_id': passage_id,
                'title': passage_metrics.get('title', f"Passage {passage_id}"),
                'overall_readability': overall_readability,
                'normalized_metrics': {
                    'lexile_score': normalized_lexile if lexile_score is not None else None,
                    'flesch_kincaid_grade': normalized_fk if fk_grade is not None else None,
                    'avg_sentence_length': normalized_asl if avg_sentence_length is not None else None
                }
            })
        return readability_scores
    
    def calculate_all_readability_metrics(self):
        """Calculate all readability metrics at once."""
        lexile_scores = self.calculate_lexile_scores()
        fk_scores = self.calculate_flesch_kincaid()
        sentence_lengths = self.calculate_avg_sentence_length()
        overall_scores = self.calculate_readability_score()
        return {
            'lexile_scores': lexile_scores,
            'flesch_kincaid_grades': fk_scores,
            'average_sentence_lengths': sentence_lengths,
            'overall_readability_scores': overall_scores
        }
    
    def compare_passages_readability(self, passage_ids=None):
        """Compare passages based on readability metrics."""
        if not self.metrics['readability']:
            logger.warning("Warning: No readability metrics calculated yet. Please calculate metrics first.")
            return []
        metrics = [m for m in self.metrics['readability'] if m.get('passage_id') in passage_ids] if passage_ids else self.metrics['readability']
        sorted_passages = sorted(metrics, key=lambda x: x.get('overall_readability', 0) if x.get('overall_readability') is not None else 0, reverse=True)
        return sorted_passages
    
    # ==================== READABILITY METRICS ====================
    
    def calculate_lexile_scores(self):
        """Calculate Lexile measures for all passages using the Eigen API."""
        lexile_scores = []
        api_url = "https://composer.api.eigen.net/api/utils/readability"
        for passage in self.passages:
            if 'text' not in passage:
                logger.warning(f"Warning No 'text' field found for passage {passage.get('id', 'unknown')}")
                continue
            text = passage['text']
            passage_id = passage.get('id', passage.get('passage_id', 'unknown'))
            try:
                response = requests.post(api_url, json={"text": text}, headers={"Content-Type": "application/json"})
                if response.status_code == 200:
                    data = response.json()
                    lexile_score = data.get('lexile', None)
                    
                    # ADDED GUARDRAIL: Cap extremely high Lexile scores at 1501
                    if lexile_score is not None and lexile_score > 2000:
                        logger.info(f"Extremely high Lexile score detected for passage {passage_id}: {lexile_score}, capping at 1501")
                        lexile_score = 1501
                        
                    score_data = {
                        'passage_id': passage_id,
                        'title': passage.get('title', f"Passage {passage_id}"),
                        'lexile_score': lexile_score
                    }
                    lexile_scores.append(score_data)
                    existing_metrics = next((item for item in self.metrics['readability'] if item.get('passage_id') == passage_id), None)
                    if existing_metrics:
                        existing_metrics['lexile_score'] = lexile_score
                    else:
                        self.metrics['readability'].append(score_data)
                else:
                    logger.error(f"API Error for passage {passage_id}: {response.status_code} - {response.text}", exc_info=True)
                    lexile_scores.append({
                        'passage_id': passage_id,
                        'title': passage.get('title', f"Passage {passage_id}"),
                        'lexile_score': None,
                        'error': f"API Error: {response.status_code}"
                    })
            except Exception as e:
                logger.error(f"Exception when calculating Lexile score for passage {passage_id}: {e}", exc_info=True)
                lexile_scores.append({
                    'passage_id': passage_id,
                    'title': passage.get('title', f"Passage {passage_id}"),
                    'lexile_score': None,
                    'error': str(e)
                })
        return lexile_scores
    
    def calculate_vocabulary_difficulty_ratio(self, oxford_wordlist_path='Oxford 5000.txt'):
        """Calculate the ratio of words beyond the 5000 most common English words."""
        try:
            with open(oxford_wordlist_path, 'r', encoding='utf-8') as f:
                common_words = set(word.strip().lower() for word in f if word.strip())
            logger.info(f"Loaded {len(common_words)} common words from Oxford 5000 list")
        except Exception as e:
            logger.error(f"Error loading Oxford 5000 word list: {e}")
            logger.info("Using an empty set as fallback - results will not be accurate")
            common_words = set()
        
        difficulty_ratios = []
        for passage in self.passages:
            if 'text' not in passage:
                logger.warning(f"Warning No 'text' field found for passage {passage.get('id', 'unknown')}")
                continue
            text = passage['text']
            passage_id = passage.get('id', passage.get('passage_id', 'unknown'))
            try:
                unique_words = set(re.findall(r'\b[a-zA-Z]+\b', text.lower()))
                if not unique_words:
                    logger.warning(f"Warning No valid words found in passage {passage_id}")
                    continue
                uncommon_words = [word for word in unique_words if word not in common_words]
                ratio = len(uncommon_words) / len(unique_words)
                uncommon_examples = uncommon_words[:10] if uncommon_words else []
                result = {
                    'passage_id': passage_id,
                    'title': passage.get('title', f"Passage {passage_id}"),
                    'vocabulary_difficulty_ratio': ratio,
                    'total_unique_words': len(unique_words),
                    'uncommon_words_count': len(uncommon_words),
                    'uncommon_examples': uncommon_examples
                }
                difficulty_ratios.append(result)
                existing_metrics = next((item for item in self.metrics['vocabulary'] if item.get('passage_id') == passage_id), None)
                if existing_metrics:
                    existing_metrics['vocabulary_difficulty_ratio'] = ratio
                else:
                    self.metrics['vocabulary'].append({
                        'passage_id': passage_id,
                        'title': passage.get('title', f"Passage {passage_id}"),
                        'vocabulary_difficulty_ratio': ratio
                    })
            except Exception as e:
                logger.error(f"Exception when calculating vocabulary difficulty for passage {passage_id}: {e}", exc_info=True)
                difficulty_ratios.append({
                    'passage_id': passage_id,
                    'title': passage.get('title', f"Passage {passage_id}"),
                    'vocabulary_difficulty_ratio': None,
                    'error': str(e)
                })
        return difficulty_ratios
    
    def calculate_academic_word_usage(self, academic_wordlist_path='Oxford Phrasal Academic Lexicon.txt'):
        """Calculate the ratio of academic words in the text."""
        try:
            with open(academic_wordlist_path, 'r', encoding='utf-8') as f:
                academic_words = set(word.strip().lower() for word in f if word.strip())
            logger.info(f"Loaded {len(academic_words)} terms from Oxford Phrasal Academic Lexicon")
        except Exception as e:
            logger.error(f"Error loading academic word list: {e}", exc_info=True)
            logger.info("Using an empty set as fallback - results will not be accurate")
            academic_words = set()
        
        academic_usage_ratios = []
        for passage in self.passages:
            if 'text' not in passage:
                logger.warning(f"Warning No 'text' field found for passage {passage.get('id', 'unknown')}")
                continue
            text = passage['text']
            passage_id = passage.get('id', passage.get('passage_id', 'unknown'))
            try:
                all_words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
                if not all_words:
                    logger.warning(f"Warning No valid words found in passage {passage_id}")
                    continue
                academic_count = sum(1 for word in all_words if word in academic_words)
                ratio = academic_count / len(all_words)
                academic_examples = list(set(word for word in all_words if word in academic_words))[:10]
                result = {
                    'passage_id': passage_id,
                    'title': passage.get('title', f"Passage {passage_id}"),
                    'academic_word_usage': ratio,
                    'total_words': len(all_words),
                    'academic_words_count': academic_count,
                    'academic_examples': academic_examples
                }
                academic_usage_ratios.append(result)
                existing_metrics = next((item for item in self.metrics['vocabulary'] if item.get('passage_id') == passage_id), None)
                if existing_metrics:
                    existing_metrics['academic_word_usage'] = ratio
                else:
                    self.metrics['vocabulary'].append({
                        'passage_id': passage_id,
                        'title': passage.get('title', f"Passage {passage_id}"),
                        'academic_word_usage': ratio
                    })
            except Exception as e:
                logger.error(f"Exception when calculating academic word usage for passage {passage_id}: {e}", exc_info=True)
                academic_usage_ratios.append({
                    'passage_id': passage_id,
                    'title': passage.get('title', f"Passage {passage_id}"),
                    'academic_word_usage': None,
                    'error': str(e)
                })
        return academic_usage_ratios
    
    def calculate_domain_specific_terminology(self, system_prompt=None):
        """
        Calculate the density of domain-specific terminology using Claude 3.7 Sonnet API.
        """
        if not hasattr(self, 'anthropic_client') or not self.anthropic_client:
            logger.error("Error: Anthropic client not initialized. Call init_anthropic_client() first.")
            return []
        
        if system_prompt is None:
            system_prompt = "You are an expert linguist specializing in vocabulary analysis. Provide accurate, objective assessments."
        
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
        
        domain_terminology_scores = []
        for passage in self.passages:
            if 'text' not in passage:
                logger.warning(f"Warning No 'text' field found for passage {passage.get('id', 'unknown')}")
                continue
            text = passage['text']
            passage_id = passage.get('id', passage.get('passage_id', 'unknown'))
            title = passage.get('title', f"Passage {passage_id}")
            try:
                prompt = PROMPT_TEMPLATE.format(passage=text)
                response = self.anthropic_client.messages.create(
                    model=self.claude_params["model"],
                    max_tokens=self.claude_params["max_tokens"],
                    temperature=self.claude_params["temperature"],
                    system=system_prompt,
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
                domain_terminology_scores.append(result_dict)
                existing_metrics = next((item for item in self.metrics['vocabulary'] if item.get('passage_id') == passage_id), None)
                if existing_metrics:
                    existing_metrics['domain_specific_terminology'] = score
                else:
                    self.metrics['vocabulary'].append({
                        'passage_id': passage_id,
                        'title': title,
                        'domain_specific_terminology': score
                    })
            except Exception as e:
                logger.error(f"Exception when calculating domain-specific terminology for passage {passage_id}: {e}", exc_info=True)
                domain_terminology_scores.append({
                    'passage_id': passage_id,
                    'title': title,
                    'domain_specific_terminology_score': None,
                    'error': str(e)
                })
        return domain_terminology_scores
    
    def calculate_subordinate_clauses(self, api_key=None):
        """
        Calculate the frequency of subordinate clauses using Claude 3.7 Sonnet API.
        """
        if not hasattr(self, 'anthropic_client') or not self.anthropic_client:
            logger.error("Error: Anthropic client not initialized. Call init_anthropic_client() first.")
            return []
        client = self.anthropic_client
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
        subordinate_clause_metrics = []
        for passage in self.passages:
            if 'text' not in passage:
                logger.warning(f"Warning No 'text' field found for passage {passage.get('id', 'unknown')}")
                continue
            text = passage['text']
            passage_id = passage.get('id', passage.get('passage_id', 'unknown'))
            title = passage.get('title', f"Passage {passage_id}")
            try:
                prompt = PROMPT_TEMPLATE.format(passage=text)
                response = client.messages.create(
                    model="claude-3-7-sonnet-20250219",
                    max_tokens=2000,
                    temperature=0,
                    system="You are an expert linguist specializing in syntax analysis. Provide accurate, objective assessments.",
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
                subordinate_clause_metrics.append(result_dict)
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
            except Exception as e:
                logger.error(f"Exception when calculating subordinate clauses for passage {passage_id}: {e}")
                subordinate_clause_metrics.append({
                    'passage_id': passage_id,
                    'title': title,
                    'complexity_score': None,
                    'error': str(e)
                })
        return subordinate_clause_metrics
    
    def calculate_syntactic_variety(self, system_prompt=None):
        """
        Analyze the variety of sentence structures using Claude API.
        """
        if not hasattr(self, 'anthropic_client') or not self.anthropic_client:
            logger.error("Error: Anthropic client not initialized. Call init_anthropic_client() first.")
            return []
        if system_prompt is None:
            system_prompt = "You are an expert linguist specializing in syntactic analysis. Provide accurate, objective assessments."
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
        syntactic_variety_metrics = []
        for passage in self.passages:
            if 'text' not in passage:
                logger.warning(f"Warning No 'text' field found for passage {passage.get('id', 'unknown')}")
                continue
            text = passage['text']
            passage_id = passage.get('id', passage.get('passage_id', 'unknown'))
            title = passage.get('title', f"Passage {passage_id}")
            try:
                prompt = PROMPT_TEMPLATE.format(passage=text)
                response = self.anthropic_client.messages.create(
                    model=self.claude_params["model"],
                    max_tokens=self.claude_params["max_tokens"],
                    temperature=self.claude_params["temperature"],
                    system=system_prompt,
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
                syntactic_variety_metrics.append(result_dict)
                existing_metrics = next((item for item in self.metrics['syntactic_complexity'] if item.get('passage_id') == passage_id), None)
                if existing_metrics:
                    existing_metrics['syntactic_variety'] = variety_score
                else:
                    self.metrics['syntactic_complexity'].append({
                        'passage_id': passage_id,
                        'title': title,
                        'syntactic_variety': variety_score
                    })
            except Exception as e:
                logger.error(f"Exception when calculating syntactic variety for passage {passage_id}: {e}")
                syntactic_variety_metrics.append({
                    'passage_id': passage_id,
                    'title': title,
                    'syntactic_variety_score': None,
                    'error': str(e)
                })
        return syntactic_variety_metrics
    
    def calculate_structural_inversions(self, system_prompt=None):
        """
        Calculate the number of structural inversions using Claude API.
        """
        if not hasattr(self, 'anthropic_client') or not self.anthropic_client:
            logger.error("Error: Anthropic client not initialized. Call init_anthropic_client() first.")
            return []
        if system_prompt is None:
            system_prompt = "You are an expert linguist specializing in sentence structure analysis. Provide accurate, objective assessments."
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
        inversion_metrics = []
        for passage in self.passages:
            if 'text' not in passage:
                logger.warning(f"Warning No 'text' field found for passage {passage.get('id', 'unknown')}")
                continue
            text = passage['text']
            passage_id = passage.get('id', passage.get('passage_id', 'unknown'))
            title = passage.get('title', f"Passage {passage_id}")
            try:
                prompt = PROMPT_TEMPLATE.format(passage=text)
                response = self.anthropic_client.messages.create(
                    model=self.claude_params["model"],
                    max_tokens=self.claude_params["max_tokens"],
                    temperature=self.claude_params["temperature"],
                    system=system_prompt,
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
                inversion_metrics.append(result_dict)
                existing_metrics = next((item for item in self.metrics['syntactic_complexity'] if item.get('passage_id') == passage_id), None)
                if existing_metrics:
                    existing_metrics['structural_inversions'] = inversion_score
                else:
                    self.metrics['syntactic_complexity'].append({
                        'passage_id': passage_id,
                        'title': title,
                        'structural_inversions': inversion_score
                    })
            except Exception as e:
                logger.error(f"Exception when calculating structural inversions for passage {passage_id}: {e}")
                inversion_metrics.append({
                    'passage_id': passage_id,
                    'title': title,
                    'inversion_score': None,
                    'error': str(e)
                })
        return inversion_metrics
    
    def calculate_embedded_clauses(self, system_prompt=None):
        """
        Calculate the frequency of embedded clauses using Claude API.
        """
        if not hasattr(self, 'anthropic_client') or not self.anthropic_client:
            logger.error("Error: Anthropic client not initialized. Call init_anthropic_client() first.")
            return []
        if system_prompt is None:
            system_prompt = "You are an expert linguist specializing in clause analysis. Provide accurate, objective assessments."
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
        embedded_clause_metrics = []
        for passage in self.passages:
            if 'text' not in passage:
                logger.warning(f"Warning No 'text' field found for passage {passage.get('id', 'unknown')}")
                continue
            text = passage['text']
            passage_id = passage.get('id', passage.get('passage_id', 'unknown'))
            title = passage.get('title', f"Passage {passage_id}")
            try:
                prompt = PROMPT_TEMPLATE.format(passage=text)
                response = self.anthropic_client.messages.create(
                    model=self.claude_params["model"],
                    max_tokens=self.claude_params["max_tokens"],
                    temperature=self.claude_params["temperature"],
                    system=system_prompt,
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
                embedded_clause_metrics.append(result_dict)
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
            except Exception as e:
                logger.error(f"Exception when calculating embedded clauses for passage {passage_id}: {e}")
                embedded_clause_metrics.append({
                    'passage_id': passage_id,
                    'title': title,
                    'embedding_score': None,
                    'error': str(e)
                })
        return embedded_clause_metrics
    
    def calculate_abstraction_level(self, system_prompt=None):
        """
        Calculate the abstraction level (concrete vs. abstract concepts ratio) using Claude API.
        """
        if not hasattr(self, 'anthropic_client') or not self.anthropic_client:
            logger.error("Error: Anthropic client not initialized. Call init_anthropic_client() first.")
            return []
        if system_prompt is None:
            system_prompt = "You are an expert in cognitive linguistics and reading comprehension. Provide accurate, objective assessments."
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
        abstraction_metrics = []
        for passage in self.passages:
            if 'text' not in passage:
                logger.warning(f"Warning No 'text' field found for passage {passage.get('id', 'unknown')}")
                continue
            text = passage['text']
            passage_id = passage.get('id', passage.get('passage_id', 'unknown'))
            title = passage.get('title', f"Passage {passage_id}")
            try:
                prompt = PROMPT_TEMPLATE.format(passage=text)
                response = self.anthropic_client.messages.create(
                    model=self.claude_params["model"],
                    max_tokens=self.claude_params["max_tokens"],
                    temperature=self.claude_params["temperature"],
                    system=system_prompt,
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
                abstraction_metrics.append(result_dict)
                existing_metrics = next((item for item in self.metrics['conceptual_density'] if item.get('passage_id') == passage_id), None)
                if existing_metrics:
                    existing_metrics['abstraction_level'] = abstraction_score
                else:
                    self.metrics['conceptual_density'].append({
                        'passage_id': passage_id,
                        'title': title,
                        'abstraction_level': abstraction_score
                    })
            except Exception as e:
                logger.error(f"Exception when calculating abstraction level for passage {passage_id}: {e}")
                abstraction_metrics.append({
                    'passage_id': passage_id,
                    'title': title,
                    'abstraction_level_score': None,
                    'error': str(e)
                })
        return abstraction_metrics
    
    def calculate_concept_familiarity(self, system_prompt=None):
        """
        Calculate concept familiarity using Claude API.
        """
        if not hasattr(self, 'anthropic_client') or not self.anthropic_client:
            logger.error("Error: Anthropic client not initialized. Call init_anthropic_client() first.")
            return []
        if system_prompt is None:
            system_prompt = "You are an expert in educational assessment and cognitive development. Provide accurate, objective assessments for high school students taking the SAT exam."
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
        familiarity_metrics = []
        for passage in self.passages:
            if 'text' not in passage:
                logger.warning(f"Warning No 'text' field found for passage {passage.get('id', 'unknown')}")
                continue
            text = passage['text']
            passage_id = passage.get('id', passage.get('passage_id', 'unknown'))
            title = passage.get('title', f"Passage {passage_id}")
            try:
                prompt = PROMPT_TEMPLATE.format(passage=text)
                response = self.anthropic_client.messages.create(
                    model=self.claude_params["model"],
                    max_tokens=self.claude_params["max_tokens"],
                    temperature=self.claude_params["temperature"],
                    system=system_prompt,
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
                familiarity_metrics.append(result_dict)
                existing_metrics = next((item for item in self.metrics['conceptual_density'] if item.get('passage_id') == passage_id), None)
                if existing_metrics:
                    existing_metrics['concept_familiarity'] = familiarity_score
                else:
                    self.metrics['conceptual_density'].append({
                        'passage_id': passage_id,
                        'title': title,
                        'concept_familiarity': familiarity_score
                    })
            except Exception as e:
                logger.error(f"Exception when calculating concept familiarity for passage {passage_id}: {e}")
                familiarity_metrics.append({
                    'passage_id': passage_id,
                    'title': title,
                    'concept_familiarity_score': None,
                    'error': str(e)
                })
        return familiarity_metrics
    
    def calculate_implied_information(self, system_prompt=None):
        """
        Calculate the amount of content requiring inference using Claude API.
        """
        if not hasattr(self, 'anthropic_client') or not self.anthropic_client:
            logger.error("Error: Anthropic client not initialized. Call init_anthropic_client() first.")
            return []
        if system_prompt is None:
            system_prompt = "You are an expert in critical reading and textual analysis. Provide accurate, objective assessments."
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
        implied_info_metrics = []
        for passage in self.passages:
            if 'text' not in passage:
                logger.warning(f"Warning No 'text' field found for passage {passage.get('id', 'unknown')}")
                continue
            text = passage['text']
            passage_id = passage.get('id', passage.get('passage_id', 'unknown'))
            title = passage.get('title', f"Passage {passage_id}")
            try:
                prompt = PROMPT_TEMPLATE.format(passage=text)
                response = self.anthropic_client.messages.create(
                    model=self.claude_params["model"],
                    max_tokens=self.claude_params["max_tokens"],
                    temperature=self.claude_params["temperature"],
                    system=system_prompt,
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
                implied_info_metrics.append(result_dict)
                existing_metrics = next((item for item in self.metrics['conceptual_density'] if item.get('passage_id') == passage_id), None)
                if existing_metrics:
                    existing_metrics['implied_information'] = implied_score
                else:
                    self.metrics['conceptual_density'].append({
                        'passage_id': passage_id,
                        'title': title,
                        'implied_information': implied_score
                    })
            except Exception as e:
                logger.error(f"Exception when calculating implied information for passage {passage_id}: {e}")
                implied_info_metrics.append({
                    'passage_id': passage_id,
                    'title': title,
                    'implied_information_score': None,
                    'error': str(e)
                })
        return implied_info_metrics
    
    def calculate_argumentative_complexity(self, system_prompt=None):
        """
        Calculate the argumentative complexity using Claude API.
        
        Parameters:
        - system_prompt: Optional custom system prompt for Claude
        
        Returns:
        - List of dictionaries with passage IDs and their argumentative complexity metrics
        """
        if not hasattr(self, 'anthropic_client') or not self.anthropic_client:
            logger.error("Error: Anthropic client not initialized. Call init_anthropic_client() first.")
            return []
        
        # Default system prompt if none provided
        if system_prompt is None:
            system_prompt = "You are an expert in rhetoric and argument analysis. Provide accurate, objective assessments."
        
        # Prompt template for Claude
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
        
        argument_metrics = []
        
        for passage in self.passages:
            # Ensure there's a text field
            if 'text' not in passage:
                logger.warning(f"Warning No 'text' field found for passage {passage.get('id', 'unknown')}")
                continue
            
            text = passage['text']
            passage_id = passage.get('id', passage.get('passage_id', 'unknown'))
            title = passage.get('title', f"Passage {passage_id}")
            
            try:
                # Prepare prompt with the passage text
                prompt = PROMPT_TEMPLATE.format(passage=text)
                
                # Call Claude API using stored parameters
                response = self.anthropic_client.messages.create(
                    model=self.claude_params["model"],
                    max_tokens=self.claude_params["max_tokens"],
                    temperature=self.claude_params["temperature"],
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                result_text = response.content[0].text
                result = extract_json_from_response(result_text)
                if result is None:
                    raise ValueError("Failed to extract valid JSON from API response.")
                
                # Extract the metrics
                main_claim = result.get('main_claim', '')
                argument_structure = result.get('argument_structure', '')
                argument_techniques = result.get('argument_techniques', [])
                complexity_score = result.get('argument_complexity_score', 0)
                explanation = result.get('explanation', '')
                
                # Create result dictionary
                result_dict = {
                    'passage_id': passage_id,
                    'title': title,
                    'main_claim': main_claim,
                    'argument_structure': argument_structure,
                    'argument_techniques': argument_techniques,
                    'argument_complexity_score': complexity_score,
                    'explanation': explanation
                }
                
                argument_metrics.append(result_dict)
                
                # Store in rhetorical structure metrics
                existing_metrics = next((item for item in self.metrics['rhetorical_structure'] 
                                    if item.get('passage_id') == passage_id), None)
                
                if existing_metrics:
                    existing_metrics['argumentative_complexity'] = complexity_score
                else:
                    self.metrics['rhetorical_structure'].append({
                        'passage_id': passage_id,
                        'title': title,
                        'argumentative_complexity': complexity_score
                    })
                
            except Exception as e:
                logger.error(f"Exception when calculating argumentative complexity for passage {passage_id}: {e}")
                argument_metrics.append({
                    'passage_id': passage_id,
                    'title': title,
                    'argument_complexity_score': None,
                    'error': str(e)
                })
        
        return argument_metrics

    def calculate_organizational_clarity(self, system_prompt=None):
        """
        Calculate the transparency of organizational structure using Claude API.
        
        Parameters:
        - system_prompt: Optional custom system prompt for Claude
        
        Returns:
        - List of dictionaries with passage IDs and their organizational clarity metrics
        """
        if not hasattr(self, 'anthropic_client') or not self.anthropic_client:
            logger.error("Error: Anthropic client not initialized. Call init_anthropic_client() first.")
            return []
        
        # Default system prompt if none provided
        if system_prompt is None:
            system_prompt = "You are an expert in text structure and composition. Provide accurate, objective assessments."
        
        # Prompt template for Claude
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
        
        organizational_metrics = []
        
        for passage in self.passages:
            # Ensure there's a text field
            if 'text' not in passage:
                logger.warning(f"Warning No 'text' field found for passage {passage.get('id', 'unknown')}")
                continue
            
            text = passage['text']
            passage_id = passage.get('id', passage.get('passage_id', 'unknown'))
            title = passage.get('title', f"Passage {passage_id}")
            
            try:
                # Prepare prompt with the passage text
                prompt = PROMPT_TEMPLATE.format(passage=text)
                
                # Call Claude API using stored parameters
                response = self.anthropic_client.messages.create(
                    model=self.claude_params["model"],
                    max_tokens=self.claude_params["max_tokens"],
                    temperature=self.claude_params["temperature"],
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                result_text = response.content[0].text
                result = extract_json_from_response(result_text)
                if result is None:
                    raise ValueError("Failed to extract valid JSON from API response.")
                
                # Extract the metrics
                organizational_pattern = result.get('organizational_pattern', '')
                structural_markers = result.get('structural_markers', [])
                paragraph_organization = result.get('paragraph_organization', '')
                clarity_score = result.get('organizational_clarity_score', 0)
                explanation = result.get('explanation', '')
                
                # Create result dictionary
                result_dict = {
                    'passage_id': passage_id,
                    'title': title,
                    'organizational_pattern': organizational_pattern,
                    'structural_markers': structural_markers,
                    'paragraph_organization': paragraph_organization,
                    'organizational_clarity_score': clarity_score,
                    'explanation': explanation
                }
                
                organizational_metrics.append(result_dict)
                
                # For difficulty measure, invert the score (5 becomes 1, 4 becomes 2, etc.)
                # since higher clarity means lower difficulty
                inverted_score = 6 - clarity_score if clarity_score is not None and clarity_score > 0 else None
                
                # Store in rhetorical structure metrics
                existing_metrics = next((item for item in self.metrics['rhetorical_structure'] 
                                    if item.get('passage_id') == passage_id), None)
                
                if existing_metrics:
                    existing_metrics['organizational_clarity'] = inverted_score
                else:
                    self.metrics['rhetorical_structure'].append({
                        'passage_id': passage_id,
                        'title': title,
                        'organizational_clarity': inverted_score
                    })
                
            except Exception as e:
                logger.error(f"Exception when calculating organizational clarity for passage {passage_id}: {e}")
                organizational_metrics.append({
                    'passage_id': passage_id,
                    'title': title,
                    'organizational_clarity_score': None,
                    'error': str(e)
                })
        
        return organizational_metrics


    def calculate_transitional_elements(self, system_prompt=None):
        """
        Calculate the clarity and sophistication of transitions using Claude API.
        
        Parameters:
        - system_prompt: Optional custom system prompt for Claude
        
        Returns:
        - List of dictionaries with passage IDs and their transitional elements metrics
        """
        if not hasattr(self, 'anthropic_client') or not self.anthropic_client:
            logger.error("Error: Anthropic client not initialized. Call init_anthropic_client() first.")
            return []
        
        # Default system prompt if none provided
        if system_prompt is None:
            system_prompt = "You are an expert in composition and textual analysis. Provide accurate, objective assessments."
        
        # Prompt template for Claude
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
        
        transition_metrics = []
        
        for passage in self.passages:
            # Ensure there's a text field
            if 'text' not in passage:
                logger.warning(f"Warning No 'text' field found for passage {passage.get('id', 'unknown')}")
                continue
            
            text = passage['text']
            passage_id = passage.get('id', passage.get('passage_id', 'unknown'))
            title = passage.get('title', f"Passage {passage_id}")
            
            try:
                # Prepare prompt with the passage text
                prompt = PROMPT_TEMPLATE.format(passage=text)
                
                # Call Claude API using stored parameters
                response = self.anthropic_client.messages.create(
                    model=self.claude_params["model"],
                    max_tokens=self.claude_params["max_tokens"],
                    temperature=self.claude_params["temperature"],
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                result_text = response.content[0].text
                result = extract_json_from_response(result_text)
                if result is None:
                    raise ValueError("Failed to extract valid JSON from API response.")
                
                # Extract the metrics
                transitional_elements = result.get('transitional_elements', [])
                transition_count = result.get('transition_count', 0)
                transition_types = result.get('transition_types', [])
                transition_score = result.get('transitional_elements_score', 0)
                explanation = result.get('explanation', '')
                
                # Create result dictionary
                result_dict = {
                    'passage_id': passage_id,
                    'title': title,
                    'transitional_elements': transitional_elements,
                    'transition_count': transition_count,
                    'transition_types': transition_types,
                    'transitional_elements_score': transition_score,
                    'explanation': explanation
                }
                
                transition_metrics.append(result_dict)
                
                # Store in rhetorical structure metrics
                existing_metrics = next((item for item in self.metrics['rhetorical_structure'] 
                                    if item.get('passage_id') == passage_id), None)
                
                if existing_metrics:
                    existing_metrics['transitional_elements'] = transition_score
                else:
                    self.metrics['rhetorical_structure'].append({
                        'passage_id': passage_id,
                        'title': title,
                        'transitional_elements': transition_score
                    })
                
            except Exception as e:
                logger.error(f"Exception when calculating transitional elements for passage {passage_id}: {e}")
                transition_metrics.append({
                    'passage_id': passage_id,
                    'title': title,
                    'transitional_elements_score': None,
                    'error': str(e)
                })
        
        return transition_metrics
    
    def calculate_prior_knowledge_requirements(self, system_prompt=None):
        """
        Calculate the background knowledge required to understand the passage using Claude API.
        
        Parameters:
        - system_prompt: Optional custom system prompt for Claude
        
        Returns:
        - List of dictionaries with passage IDs and their prior knowledge requirement metrics
        """
        if not hasattr(self, 'anthropic_client') or not self.anthropic_client:
            logger.error("Error: Anthropic client not initialized. Call init_anthropic_client() first.")
            return []
        
        # Default system prompt if none provided
        if system_prompt is None:
            system_prompt = "You are an expert in educational assessment and content analysis. Provide accurate, objective assessments for high school students preparing for the SAT."
        
        # Prompt template for Claude
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
        
        prior_knowledge_metrics = []
        
        for passage in self.passages:
            # Ensure there's a text field
            if 'text' not in passage:
                logger.warning(f"Warning No 'text' field found for passage {passage.get('id', 'unknown')}")
                continue
            
            text = passage['text']
            passage_id = passage.get('id', passage.get('passage_id', 'unknown'))
            title = passage.get('title', f"Passage {passage_id}")
            
            try:
                # Prepare prompt with the passage text
                prompt = PROMPT_TEMPLATE.format(passage=text)
                
                # Call Claude API using stored parameters
                response = self.anthropic_client.messages.create(
                    model=self.claude_params["model"],
                    max_tokens=self.claude_params["max_tokens"],
                    temperature=self.claude_params["temperature"],
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                result_text = response.content[0].text
                result = extract_json_from_response(result_text)
                if result is None:
                    raise ValueError("Failed to extract valid JSON from API response.")
                
                # Extract the metrics
                required_knowledge = result.get('required_knowledge', [])
                knowledge_domains = result.get('knowledge_domains', [])
                prior_knowledge_score = result.get('prior_knowledge_score', 0)
                explanation = result.get('explanation', '')
                
                # Create result dictionary
                result_dict = {
                    'passage_id': passage_id,
                    'title': title,
                    'required_knowledge': required_knowledge,
                    'knowledge_domains': knowledge_domains,
                    'prior_knowledge_score': prior_knowledge_score,
                    'explanation': explanation
                }
                
                prior_knowledge_metrics.append(result_dict)
                
                # Store in content accessibility metrics
                existing_metrics = next((item for item in self.metrics['content_accessibility'] 
                                    if item.get('passage_id') == passage_id), None)
                
                if existing_metrics:
                    existing_metrics['prior_knowledge_requirements'] = prior_knowledge_score
                else:
                    self.metrics['content_accessibility'].append({
                        'passage_id': passage_id,
                        'title': title,
                        'prior_knowledge_requirements': prior_knowledge_score
                    })
                
            except Exception as e:
                logger.error(f"Exception when calculating prior knowledge requirements for passage {passage_id}: {e}")
                prior_knowledge_metrics.append({
                    'passage_id': passage_id,
                    'title': title,
                    'prior_knowledge_score': None,
                    'error': str(e)
                })
        
        return prior_knowledge_metrics

    def calculate_disciplinary_perspective(self, system_prompt=None):
        """
        Calculate the field-specific thought patterns required in the passage using Claude API.
        
        Parameters:
        - system_prompt: Optional custom system prompt for Claude
        
        Returns:
        - List of dictionaries with passage IDs and their disciplinary perspective metrics
        """
        if not hasattr(self, 'anthropic_client') or not self.anthropic_client:
            logger.error("Error: Anthropic client not initialized. Call init_anthropic_client() first.")
            return []
        
        # Default system prompt if none provided
        if system_prompt is None:
            system_prompt = "You are an expert in academic disciplines and educational assessment. Provide accurate, objective assessments for high school students."
        
        # Prompt template for Claude
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
        
        disciplinary_metrics = []
        
        for passage in self.passages:
            # Ensure there's a text field
            if 'text' not in passage:
                logger.warning(f"Warning No 'text' field found for passage {passage.get('id', 'unknown')}")
                continue
            
            text = passage['text']
            passage_id = passage.get('id', passage.get('passage_id', 'unknown'))
            title = passage.get('title', f"Passage {passage_id}")
            
            try:
                # Prepare prompt with the passage text
                prompt = PROMPT_TEMPLATE.format(passage=text)
                
                # Call Claude API using stored parameters
                response = self.anthropic_client.messages.create(
                    model=self.claude_params["model"],
                    max_tokens=self.claude_params["max_tokens"],
                    temperature=self.claude_params["temperature"],
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                result_text = response.content[0].text
                result = extract_json_from_response(result_text)
                if result is None:
                    raise ValueError("Failed to extract valid JSON from API response.")
                
                # Extract the metrics
                primary_disciplines = result.get('primary_disciplines', [])
                field_specific_elements = result.get('field_specific_elements', [])
                disciplinary_score = result.get('disciplinary_perspective_score', 0)
                explanation = result.get('explanation', '')
                
                # Create result dictionary
                result_dict = {
                    'passage_id': passage_id,
                    'title': title,
                    'primary_disciplines': primary_disciplines,
                    'field_specific_elements': field_specific_elements,
                    'disciplinary_perspective_score': disciplinary_score,
                    'explanation': explanation
                }
                
                disciplinary_metrics.append(result_dict)
                
                # Store in content accessibility metrics
                existing_metrics = next((item for item in self.metrics['content_accessibility'] 
                                    if item.get('passage_id') == passage_id), None)
                
                if existing_metrics:
                    existing_metrics['disciplinary_perspective'] = disciplinary_score
                else:
                    self.metrics['content_accessibility'].append({
                        'passage_id': passage_id,
                        'title': title,
                        'disciplinary_perspective': disciplinary_score
                    })
                
            except Exception as e:
                logger.error(f"Exception when calculating disciplinary perspective for passage {passage_id}: {e}")
                disciplinary_metrics.append({
                    'passage_id': passage_id,
                    'title': title,
                    'disciplinary_perspective_score': None,
                    'error': str(e)
                })
        
        return disciplinary_metrics


    def calculate_language_modernity(self, system_prompt=None):
        """
        Calculate the contemporary vs. archaic nature of language in the passage using Claude API.
        
        Parameters:
        - system_prompt: Optional custom system prompt for Claude
        
        Returns:
        - List of dictionaries with passage IDs and their language modernity metrics
        """
        if not hasattr(self, 'anthropic_client') or not self.anthropic_client:
            logger.error("Error: Anthropic client not initialized. Call init_anthropic_client() first.")
            return []
        
        # Default system prompt if none provided
        if system_prompt is None:
            system_prompt = "You are an expert in linguistic analysis and historical language development. Provide accurate, objective assessments for high school students."
        
        # Prompt template for Claude
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
        
        language_modernity_metrics = []
        
        for passage in self.passages:
            # Ensure there's a text field
            if 'text' not in passage:
                logger.warning(f"Warning No 'text' field found for passage {passage.get('id', 'unknown')}")
                continue
            
            text = passage['text']
            passage_id = passage.get('id', passage.get('passage_id', 'unknown'))
            title = passage.get('title', f"Passage {passage_id}")
            
            try:
                # Prepare prompt with the passage text
                prompt = PROMPT_TEMPLATE.format(passage=text)
                
                # Call Claude API using stored parameters
                response = self.anthropic_client.messages.create(
                    model=self.claude_params["model"],
                    max_tokens=self.claude_params["max_tokens"],
                    temperature=self.claude_params["temperature"],
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                result_text = response.content[0].text
                result = extract_json_from_response(result_text)
                if result is None:
                    raise ValueError("Failed to extract valid JSON from API response.")
                
                # Extract the metrics
                archaic_elements = result.get('archaic_elements', [])
                approximate_period = result.get('approximate_period', '')
                modernity_score = result.get('language_modernity_score', 0)
                explanation = result.get('explanation', '')
                
                # Create result dictionary
                result_dict = {
                    'passage_id': passage_id,
                    'title': title,
                    'archaic_elements': archaic_elements,
                    'approximate_period': approximate_period,
                    'language_modernity_score': modernity_score,
                    'explanation': explanation
                }
                
                language_modernity_metrics.append(result_dict)
                
                # Store in content accessibility metrics
                existing_metrics = next((item for item in self.metrics['content_accessibility'] 
                                    if item.get('passage_id') == passage_id), None)
                
                if existing_metrics:
                    existing_metrics['language_modernity'] = modernity_score
                else:
                    self.metrics['content_accessibility'].append({
                        'passage_id': passage_id,
                        'title': title,
                        'language_modernity': modernity_score
                    })
                
            except Exception as e:
                logger.error(f"Exception when calculating language modernity for passage {passage_id}: {e}")
                language_modernity_metrics.append({
                    'passage_id': passage_id,
                    'title': title,
                    'language_modernity_score': None,
                    'error': str(e)
                })
        
        return language_modernity_metrics
    
    def calculate_inference_requirement(self, system_prompt=None):
        """
        Calculate the level of inference required to understand the passage using Claude API.
        
        Parameters:
        - system_prompt: Optional custom system prompt for Claude
        
        Returns:
        - List of dictionaries with passage IDs and their inference requirement metrics
        """
        if not hasattr(self, 'anthropic_client') or not self.anthropic_client:
            logger.error("Error: Anthropic client not initialized. Call init_anthropic_client() first.")
            return []
        
        # Default system prompt if none provided
        if system_prompt is None:
            system_prompt = "You are an expert in reading comprehension and cognitive analysis. Provide accurate, objective assessments for high school students taking the SAT."
        
        # Prompt template for Claude
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
        
        inference_metrics = []
        
        for passage in self.passages:
            # Ensure there's a text field
            if 'text' not in passage:
                logger.warning(f"Warning No 'text' field found for passage {passage.get('id', 'unknown')}")
                continue
            
            text = passage['text']
            passage_id = passage.get('id', passage.get('passage_id', 'unknown'))
            title = passage.get('title', f"Passage {passage_id}")
            
            try:
                # Prepare prompt with the passage text
                prompt = PROMPT_TEMPLATE.format(passage=text)
                
                # Call Claude API using stored parameters
                response = self.anthropic_client.messages.create(
                    model=self.claude_params["model"],
                    max_tokens=self.claude_params["max_tokens"],
                    temperature=self.claude_params["temperature"],
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                result_text = response.content[0].text
                result = extract_json_from_response(result_text)
                if result is None:
                    raise ValueError("Failed to extract valid JSON from API response.")
                
                # Extract the metrics
                inference_examples = result.get('inference_examples', [])
                inference_types = result.get('inference_types', [])
                inference_score = result.get('inference_requirement_score', 0)
                explanation = result.get('explanation', '')
                
                # Create result dictionary
                result_dict = {
                    'passage_id': passage_id,
                    'title': title,
                    'inference_examples': inference_examples,
                    'inference_types': inference_types,
                    'inference_requirement_score': inference_score,
                    'explanation': explanation
                }
                
                inference_metrics.append(result_dict)
                
                # Store in cognitive demands metrics
                existing_metrics = next((item for item in self.metrics['cognitive_demands'] 
                                    if item.get('passage_id') == passage_id), None)
                
                if existing_metrics:
                    existing_metrics['inference_requirement'] = inference_score
                else:
                    self.metrics['cognitive_demands'].append({
                        'passage_id': passage_id,
                        'title': title,
                        'inference_requirement': inference_score
                    })
                
            except Exception as e:
                logger.error(f"Exception when calculating inference requirement for passage {passage_id}: {e}")
                inference_metrics.append({
                    'passage_id': passage_id,
                    'title': title,
                    'inference_requirement_score': None,
                    'error': str(e)
                })
        
        return inference_metrics

    def calculate_figurative_language(self, system_prompt=None):
        """
        Calculate the density of figurative language using Claude API.
        
        Parameters:
        - system_prompt: Optional custom system prompt for Claude
        
        Returns:
        - List of dictionaries with passage IDs and their figurative language metrics
        """
        if not hasattr(self, 'anthropic_client') or not self.anthropic_client:
            logger.error("Error: Anthropic client not initialized. Call init_anthropic_client() first.")
            return []
        
        # Default system prompt if none provided
        if system_prompt is None:
            system_prompt = "You are an expert in literary analysis and rhetoric. Provide accurate, objective assessments for high school students."
        
        # Prompt template for Claude
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
        
        figurative_language_metrics = []
        
        for passage in self.passages:
            # Ensure there's a text field
            if 'text' not in passage:
                logger.warning(f"Warning No 'text' field found for passage {passage.get('id', 'unknown')}")
                continue
            
            text = passage['text']
            passage_id = passage.get('id', passage.get('passage_id', 'unknown'))
            title = passage.get('title', f"Passage {passage_id}")
            
            try:
                # Prepare prompt with the passage text
                prompt = PROMPT_TEMPLATE.format(passage=text)
                
                # Call Claude API using stored parameters
                response = self.anthropic_client.messages.create(
                    model=self.claude_params["model"],
                    max_tokens=self.claude_params["max_tokens"],
                    temperature=self.claude_params["temperature"],
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                result_text = response.content[0].text
                result = extract_json_from_response(result_text)
                if result is None:
                    raise ValueError("Failed to extract valid JSON from API response.")
                
                # Extract the metrics
                figurative_language = result.get('figurative_language', [])
                device_types = result.get('device_types', [])
                device_count = result.get('device_count', 0)
                figurative_score = result.get('figurative_language_score', 0)
                explanation = result.get('explanation', '')
                
                # Create result dictionary
                result_dict = {
                    'passage_id': passage_id,
                    'title': title,
                    'figurative_language': figurative_language,
                    'device_types': device_types,
                    'device_count': device_count,
                    'figurative_language_score': figurative_score,
                    'explanation': explanation
                }
                
                figurative_language_metrics.append(result_dict)
                
                # Store in cognitive demands metrics
                existing_metrics = next((item for item in self.metrics['cognitive_demands'] 
                                    if item.get('passage_id') == passage_id), None)
                
                if existing_metrics:
                    existing_metrics['figurative_language'] = figurative_score
                else:
                    self.metrics['cognitive_demands'].append({
                        'passage_id': passage_id,
                        'title': title,
                        'figurative_language': figurative_score
                    })
                
            except Exception as e:
                logger.error(f"Exception when calculating figurative language for passage {passage_id}: {e}")
                figurative_language_metrics.append({
                    'passage_id': passage_id,
                    'title': title,
                    'figurative_language_score': None,
                    'error': str(e)
                })
        
        return figurative_language_metrics

    def calculate_authors_purpose(self, system_prompt=None):
        """
        Calculate the clarity vs. obscurity of author's intent using Claude API.
        
        Parameters:
        - system_prompt: Optional custom system prompt for Claude
        
        Returns:
        - List of dictionaries with passage IDs and their author's purpose metrics
        """
        if not hasattr(self, 'anthropic_client') or not self.anthropic_client:
            logger.error("Error: Anthropic client not initialized. Call init_anthropic_client() first.")
            return []
        
        # Default system prompt if none provided
        if system_prompt is None:
            system_prompt = "You are an expert in rhetorical analysis and textual interpretation. Provide accurate, objective assessments for high school students."
        
        # Prompt template for Claude
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
        
        authors_purpose_metrics = []
        
        for passage in self.passages:
            # Ensure there's a text field
            if 'text' not in passage:
                logger.warning(f"Warning No 'text' field found for passage {passage.get('id', 'unknown')}")
                continue
            
            text = passage['text']
            passage_id = passage.get('id', passage.get('passage_id', 'unknown'))
            title = passage.get('title', f"Passage {passage_id}")
            
            try:
                # Prepare prompt with the passage text
                prompt = PROMPT_TEMPLATE.format(passage=text)
                
                # Call Claude API using stored parameters
                response = self.anthropic_client.messages.create(
                    model=self.claude_params["model"],
                    max_tokens=self.claude_params["max_tokens"],
                    temperature=self.claude_params["temperature"],
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                result_text = response.content[0].text
                result = extract_json_from_response(result_text)
                if result is None:
                    raise ValueError("Failed to extract valid JSON from API response.")
                
                # Extract the metrics
                apparent_purposes = result.get('apparent_purposes', [])
                purpose_indicators = result.get('purpose_indicators', [])
                purpose_score = result.get('authors_purpose_score', 0)
                explanation = result.get('explanation', '')
                
                # Create result dictionary
                result_dict = {
                    'passage_id': passage_id,
                    'title': title,
                    'apparent_purposes': apparent_purposes,
                    'purpose_indicators': purpose_indicators,
                    'authors_purpose_score': purpose_score,
                    'explanation': explanation
                }
                
                authors_purpose_metrics.append(result_dict)
                
                # Store in cognitive demands metrics
                existing_metrics = next((item for item in self.metrics['cognitive_demands'] 
                                    if item.get('passage_id') == passage_id), None)
                
                if existing_metrics:
                    existing_metrics['authors_purpose'] = purpose_score
                else:
                    self.metrics['cognitive_demands'].append({
                        'passage_id': passage_id,
                        'title': title,
                        'authors_purpose': purpose_score
                    })
                
            except Exception as e:
                logger.error(f"Exception when calculating author's purpose for passage {passage_id}: {e}")
                authors_purpose_metrics.append({
                    'passage_id': passage_id,
                    'title': title,
                    'authors_purpose_score': None,
                    'error': str(e)
                })
        
        return authors_purpose_metrics
    
    def export_results_to_json(self, filename='passage_analysis_results.json'):
        """
        Export all analysis results to a JSON file.
        
        Parameters:
        - filename: Name of the JSON file to create
        
        Returns:
        - Path to the created JSON file
        """
        import json
        import os
        
        # Prepare the complete results dictionary
        results = {
            'passages': self.passages,
            'metrics': self.metrics,
            'analysis_date': self._get_current_datetime(),
            'overall_difficulty': self.calculate_overall_difficulty()
        }
        
        # Convert to JSON and save to file
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results successfully saved to {os.path.abspath(filename)}")
            return os.path.abspath(filename)
        except Exception as e:
            logger.error(f"Error saving results to JSON: {e}")
            return None

    def export_summary_to_csv(self, filename='passage_difficulty_summary.csv'):
        """
        Export a condensed summary of difficulty scores to a CSV file.
        
        Parameters:
        - filename: Name of the CSV file to create
        
        Returns:
        - Path to the created CSV file
        """
        import csv
        import os
        
        # Get overall difficulty scores
        difficulty_scores = self.calculate_overall_difficulty()
        
        # Define difficulty interpretation thresholds
        def interpret_difficulty(score):
            if score is None:
                return "Unknown"
            # Updated thresholds for 1-10 scale
            elif score < 3:
                return "Very Easy"
            elif score < 5:
                return "Easy"
            elif score < 7:
                return "Medium"
            elif score < 9:
                return "Hard"
            else:
                return "Very Hard"
        
        # Prepare CSV data
        csv_data = [["passage_id", "title", "overall_score", "difficulty_level"]]
        
        for score in difficulty_scores:
            row = [
                score.get('passage_id', 'unknown'),
                score.get('title', f"Passage {score.get('passage_id', 'unknown')}"),
                round(score.get('overall_score', 0), 2),
                interpret_difficulty(score.get('overall_score'))
            ]
            csv_data.append(row)
        
        # Write to CSV file
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(csv_data)
            logger.info(f"Summary successfully saved to {os.path.abspath(filename)}")
            return os.path.abspath(filename)
        except Exception as e:
            logger.error(f"Error saving summary to CSV: {e}")
            return None
        
    def _normalize_vocabulary_ratio(self, ratio):
        """
        Normalize vocabulary ratio metrics (vocabulary_difficulty_ratio and academic_word_usage)
        to a 1-5 scale based on specified thresholds.
        
        Parameters:
        - ratio: The ratio value to normalize
        
        Returns:
        - Normalized score on a 1-5 scale
        """
        if ratio is None:
            return None
        
        if ratio < 0.2:
            return 1
        elif ratio < 0.4:
            return 2
        elif ratio < 0.6:
            return 3
        elif ratio < 0.8:
            return 4
        else:
            return 5

    def calculate_vocabulary_score(self):
        """
        Calculate a combined vocabulary score based on all vocabulary metrics,
        normalizing ratio metrics appropriately.
        
        Returns:
        - List of dictionaries with passage IDs and their vocabulary scores
        """
        vocabulary_scores = []
        
        if not self.metrics['vocabulary']:
            logger.warning("Warning: No vocabulary metrics calculated yet. Please calculate metrics first.")
            return []
        
        for passage_metrics in self.metrics['vocabulary']:
            passage_id = passage_metrics.get('passage_id')
            difficulty_ratio = passage_metrics.get('vocabulary_difficulty_ratio')
            academic_usage = passage_metrics.get('academic_word_usage')
            domain_specific = passage_metrics.get('domain_specific_terminology')
            
            # Normalize ratio metrics to 1-5 scale
            normalized_difficulty = self._normalize_vocabulary_ratio(difficulty_ratio)
            normalized_academic = self._normalize_vocabulary_ratio(academic_usage)
            
            # Collect all available normalized scores
            normalized_scores = []
            if normalized_difficulty is not None:
                normalized_scores.append(normalized_difficulty)
            if normalized_academic is not None:
                normalized_scores.append(normalized_academic)
            if domain_specific is not None:
                normalized_scores.append(domain_specific)
            
            # Calculate overall vocabulary score
            overall_vocabulary = sum(normalized_scores) / len(normalized_scores) if normalized_scores else None
            
            # Add to passage metrics
            passage_metrics['overall_vocabulary'] = overall_vocabulary
            
            vocabulary_scores.append({
                'passage_id': passage_id,
                'title': passage_metrics.get('title', f"Passage {passage_id}"),
                'overall_vocabulary': overall_vocabulary,
                'normalized_metrics': {
                    'vocabulary_difficulty_ratio': normalized_difficulty,
                    'academic_word_usage': normalized_academic,
                    'domain_specific_terminology': domain_specific
                }
            })
        
        return vocabulary_scores
        
    def get_normalized_category_score(self, category, passage_id):
        """
        Get a normalized score (1-5 scale) for a specific category and passage.
        
        Parameters:
        - category: Category name from self.metrics
        - passage_id: ID of the passage
        
        Returns:
        - Normalized score on a 1-5 scale
        """
        metrics = next((item for item in self.metrics[category] if item.get('passage_id') == passage_id), None)
        
        if not metrics:
            return None
            
        # Remove non-score fields
        score_fields = {k: v for k, v in metrics.items() 
                    if k not in ['passage_id', 'title', 'explanation'] 
                    and isinstance(v, (int, float))}
        
        # For readability category, we need special handling
        if category == 'readability':
            # If overall_readability is already calculated, use it
            if 'overall_readability' in metrics and metrics['overall_readability'] is not None:
                return metrics['overall_readability']
                
            # Otherwise, normalize individual metrics
            normalized_scores = []
            
            if 'lexile_score' in metrics and metrics['lexile_score'] is not None:
                lexile = metrics['lexile_score']
                if lexile < 900:
                    normalized_scores.append(1)
                elif lexile < 1200:
                    normalized_scores.append(2)
                elif lexile < 1300:
                    normalized_scores.append(3)
                elif lexile < 1400:
                    normalized_scores.append(4)
                else:
                    normalized_scores.append(5)
                    
            if 'flesch_kincaid_grade' in metrics and metrics['flesch_kincaid_grade'] is not None:
                fk = metrics['flesch_kincaid_grade']
                if fk < 9:
                    normalized_scores.append(1)
                elif fk < 11:
                    normalized_scores.append(2)
                elif fk < 13:
                    normalized_scores.append(3)
                elif fk < 15:
                    normalized_scores.append(4)
                else:
                    normalized_scores.append(5)
                    
            if 'avg_sentence_length' in metrics and metrics['avg_sentence_length'] is not None:
                asl = metrics['avg_sentence_length']
                if asl < 18:
                    normalized_scores.append(1)
                elif asl < 22:
                    normalized_scores.append(2)
                elif asl < 25:
                    normalized_scores.append(3)
                elif asl < 30:
                    normalized_scores.append(4)
                else:
                    normalized_scores.append(5)
                    
            if normalized_scores:
                return sum(normalized_scores) / len(normalized_scores)
            else:
                return None
        
        # For other categories, simply average the existing scores (which should already be on a 1-5 scale)
        if score_fields:
            return sum(score_fields.values()) / len(score_fields)
        else:
            return None

    def calculate_overall_difficulty(self):
        """
        Calculate an overall difficulty score for each passage based on all metrics.
        
        Returns:
        - List of dictionaries with passage IDs and overall difficulty scores (1-10 scale)
        """
        # Collect all passages with at least some metrics calculated
        passage_ids = set()
        for category in self.metrics.values():
            for item in category:
                if 'passage_id' in item:
                    passage_ids.add(item['passage_id'])
        
        # First, ensure vocabulary and readability scores are calculated with proper normalization
        self.calculate_readability_score()
        self.calculate_vocabulary_score()
        
        # Define category weights
        weights = {
            'readability': 0.10,
            'vocabulary': 0.15,
            'syntactic_complexity': 0.15,
            'conceptual_density': 0.15,
            'rhetorical_structure': 0.15,
            'content_accessibility': 0.15,
            'cognitive_demands': 0.15
        }
        
        overall_scores = []
        
        for passage_id in passage_ids:
            # Get title for the passage
            title = None
            for passage in self.passages:
                if passage.get('id') == passage_id or passage.get('passage_id') == passage_id:
                    title = passage.get('title', f"Passage {passage_id}")
                    break
            
            # If not found in passages, try to get from metrics
            if title is None:
                for category in self.metrics.values():
                    for item in category:
                        if item.get('passage_id') == passage_id and 'title' in item:
                            title = item['title']
                            break
                    if title:
                        break
            
            # If still not found, use default
            if title is None:
                title = f"Passage {passage_id}"
            
            # Collect scores from each category
            category_scores = {}

            for category, metrics_list in self.metrics.items():
                # Find metrics for this passage
                metrics = next((m for m in metrics_list if m.get('passage_id') == passage_id), None)
                
                if metrics:
                    if category == 'vocabulary' and 'overall_vocabulary' in metrics:
                        # Use our pre-calculated normalized vocabulary score
                        category_scores[category] = metrics['overall_vocabulary']
                    elif category == 'readability' and 'overall_readability' in metrics:
                        # Use our pre-calculated normalized readability score
                        category_scores[category] = metrics['overall_readability']
                    else:
                        # Remove non-score fields and use the average for other categories
                        score_fields = {k: v for k, v in metrics.items() 
                                    if k not in ['passage_id', 'title', 'explanation', 'overall_vocabulary', 'overall_readability'] 
                                    and isinstance(v, (int, float))}
                        
                        # Calculate average score for this category
                        if score_fields:
                            category_scores[category] = sum(score_fields.values()) / len(score_fields)
            
            # Calculate weighted overall score
            if category_scores:
                weighted_sum = 0
                total_weight = 0
                
                for category, score in category_scores.items():
                    weight = weights.get(category, 0)
                    weighted_sum += score * weight
                    total_weight += weight
                
                # Scale to 1-10
                if total_weight > 0:
                    overall_score = (weighted_sum / total_weight) * 2  # Convert 1-5 scale to 1-10
                else:
                    overall_score = None
            else:
                overall_score = None
            
            overall_scores.append({
                'passage_id': passage_id,
                'title': title,
                'overall_score': overall_score,
                'category_scores': category_scores
            })
        
        # Sort by difficulty (highest to lowest)
        overall_scores.sort(
            key=lambda x: (x['overall_score'] is None, -1 * (x['overall_score'] or 0))
        )
        
        return overall_scores

    def _get_current_datetime(self):
        """Get current date and time as a formatted string."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def _save_metric(self, metric_name, data):
        """
        Save the provided data to a JSON file named {metric_name}.json.
        """
        with open(f"{metric_name}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def evaluate_passages(self, api_key=None):
        """
        Run a complete evaluation of all passages, calculating all metrics.
        
        Parameters:
        - api_key: API key for Claude (if not provided, will look for ANTHROPIC_API_KEY env variable)
        
        Returns:
        - Dictionary with overall results
        """
        # Initialize Claude client if needed for qualitative metrics
        if api_key:
            self.init_anthropic_client(api_key)
        
        logger.info("Starting comprehensive passage evaluation...")
        
        # Calculate readability metrics
        logger.info("\n=== Calculating Readability Metrics ===")
        try:
            lexile = self.calculate_lexile_scores()
            self._save_metric("lexile_scores", lexile)
        except Exception as e:
            logger.error(f"Error calculating Lexile scores: {e}")
        fk = self.calculate_flesch_kincaid()
        self._save_metric("flesch_kincaid", fk)
        asl = self.calculate_avg_sentence_length()
        self._save_metric("avg_sentence_length", asl)
        
        # Calculate vocabulary metrics
        logger.info("\n=== Calculating Vocabulary Metrics ===")
        try:
            vocab_difficulty = self.calculate_vocabulary_difficulty_ratio()
            self._save_metric("vocabulary_difficulty_ratio", vocab_difficulty)
        except Exception as e:
            logger.error(f"Error calculating vocabulary difficulty: {e}")
        try:
            academic_usage = self.calculate_academic_word_usage()
            self._save_metric("academic_word_usage", academic_usage)
        except Exception as e:
            logger.error(f"Error calculating academic word usage: {e}")
        
        # Calculate remaining metrics using Claude (if available)
        if hasattr(self, 'anthropic_client') and self.anthropic_client:
            # Domain-specific terminology
            logger.info("\n=== Calculating Domain-Specific Terminology ===")
            domain_specific = self.calculate_domain_specific_terminology()
            self._save_metric("domain_specific_terminology", domain_specific)
            
            # Syntactic complexity
            logger.info("\n=== Calculating Syntactic Complexity Metrics ===")
            subordinate = self.calculate_subordinate_clauses()
            self._save_metric("subordinate_clauses", subordinate)
            variety = self.calculate_syntactic_variety()
            self._save_metric("syntactic_variety", variety)
            inversions = self.calculate_structural_inversions()
            self._save_metric("structural_inversions", inversions)
            embedded = self.calculate_embedded_clauses()
            self._save_metric("embedded_clauses", embedded)
            
            # Conceptual density
            logger.info("\n=== Calculating Conceptual Density Metrics ===")
            abstraction = self.calculate_abstraction_level()
            self._save_metric("abstraction_level", abstraction)
            familiarity = self.calculate_concept_familiarity()
            self._save_metric("concept_familiarity", familiarity)
            implied = self.calculate_implied_information()
            self._save_metric("implied_information", implied)
            
            # Rhetorical structure
            logger.info("\n=== Calculating Rhetorical Structure Metrics ===")
            argumentative = self.calculate_argumentative_complexity()
            self._save_metric("argumentative_complexity", argumentative)
            organizational = self.calculate_organizational_clarity()
            self._save_metric("organizational_clarity", organizational)
            transitional = self.calculate_transitional_elements()
            self._save_metric("transitional_elements", transitional)
            
            # Content accessibility
            logger.info("\n=== Calculating Content Accessibility Metrics ===")
            prior_knowledge = self.calculate_prior_knowledge_requirements()
            self._save_metric("prior_knowledge_requirements", prior_knowledge)
            disciplinary = self.calculate_disciplinary_perspective()
            self._save_metric("disciplinary_perspective", disciplinary)
            language_modernity = self.calculate_language_modernity()
            self._save_metric("language_modernity", language_modernity)
            
            # Cognitive demands
            logger.info("\n=== Calculating Cognitive Demand Metrics ===")
            inference = self.calculate_inference_requirement()
            self._save_metric("inference_requirement", inference)
            figurative = self.calculate_figurative_language()
            self._save_metric("figurative_language", figurative)
            authors = self.calculate_authors_purpose()
            self._save_metric("authors_purpose", authors)
        else:
            logger.warning("\nWarning: Claude API client not initialized. Skipping qualitative metrics.")
            logger.info("To calculate all metrics, please provide an API key or set the ANTHROPIC_API_KEY environment variable.")
        
        # Calculate overall scores
        logger.info("\n=== Calculating Overall Difficulty Scores ===")
        overall_scores = self.calculate_overall_difficulty()
        
        logger.info("\nEvaluation complete!")
        logger.info(f"Analyzed {len(self.passages)} passages across {len(self.metrics)} metric categories.")
        
        logger.info("\n=== Difficulty Ranking (Most to Least Difficult) ===")
        for i, score in enumerate(overall_scores):
            difficulty_level = "Unknown"
            if score['overall_score'] is not None:
                if score['overall_score'] < 90:
                    difficulty_level = "Very Easy"
                elif score['overall_score'] < 120:
                    difficulty_level = "Easy"
                elif score['overall_score'] < 150:
                    difficulty_level = "Medium"
                elif score['overall_score'] < 200:
                    difficulty_level = "Hard"
                else:
                    difficulty_level = "Very Hard"
            
            logger.info(f"{i+1}. {score['title']} (ID: {score['passage_id']}): " + 
                  (f"{score['overall_score']:.2f}/10 - {difficulty_level}" if score['overall_score'] is not None else "Not enough data"))
        
        return {
            'passages': self.passages,
            'metrics': self.metrics,
            'overall_scores': overall_scores
        }

    def generate_report(self, output_dir='.'):
        """
        Generate comprehensive reports of the analysis results.
        
        Parameters:
        - output_dir: Directory to save the report files
        
        Returns:
        - Dictionary with paths to generated files
        """
        import os
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Define file paths
        json_path = os.path.join(output_dir, 'passage_analysis_results.json')
        csv_path = os.path.join(output_dir, 'passage_difficulty_summary.csv')
        
        # Calculate overall scores if not already done
        if not hasattr(self, 'overall_scores'):
            self.overall_scores = self.calculate_overall_difficulty()
        
        # Export full results to JSON
        json_file = self.export_results_to_json(json_path)
        
        # Export summary to CSV
        csv_file = self.export_summary_to_csv(csv_path)
        
        return {
            'json_report': json_file,
            'csv_summary': csv_file
        }
