"""
Question Difficulty Assessment Framework

This framework evaluates the difficulty of educational questions based on multiple dimensions:
- Knowledge Requirements (30%)
- Cognitive Complexity (30%)
- Context and Ambiguity (15%)
- Question Structure and Format (5%)
- Linguistic and Semantic Factors (15%)
- Learning Objectives and Assessment (5%)

Input: JSON file with questions containing passage, question, options, target_grade
Output: Detailed difficulty assessment and overall difficulty rating
"""

# Required imports
import json
import os
import re
import statistics
import anthropic
import functools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from tqdm import tqdm
try:
    from cache_manager import cache_llm_response, get_cache
    CACHE_AVAILABLE = True
except ImportError:
    # Define a no-op decorator if cache_manager is not available
    def cache_llm_response(func):
        return func
    CACHE_AVAILABLE = False

# Load environment variables for API keys
load_dotenv()

# Constants
DIFFICULTY_LEVELS = {
    (0.00, 0.20): "Very Easy",
    (0.21, 0.40): "Easy",
    (0.41, 0.60): "Moderate",
    (0.61, 0.80): "Hard",
    (0.81, 1.00): "Very Hard"
}

DIMENSION_WEIGHTS = {
    "knowledge_requirements": 0.30,
    "cognitive_complexity": 0.30,
    "context_and_ambiguity": 0.15,
    "question_structure": 0.05,
    "linguistic_factors": 0.05,
    "learning_objectives": 0.15
}

# Knowledge Requirements sub-weights
KNOWLEDGE_WEIGHTS = {
    "prior_knowledge": 0.25,
    "domain_specific": 0.20,
    "knowledge_level": 0.25,
    "concept_abstractness": 0.15,
    "specialized_terminology": 0.15
}

# Cognitive Complexity sub-weights
COGNITIVE_WEIGHTS = {
    "thinking_requirements": 0.30,
    "cognitive_level": 0.30,
    "problem_complexity": 0.20,
    "cognitive_bias": 0.20
}

# Context and Ambiguity sub-weights
CONTEXT_WEIGHTS = {
    "context_dependency": 0.30,
    "context_clarity": 0.30,
    "context_difficulty": 0.40
}

# Question Structure sub-weights
STRUCTURE_WEIGHTS = {
    "question_format": 0.40,
    "visual_aids": 0.30,
    "question_clarity": 0.30
}

# Linguistic Factors sub-weights
LINGUISTIC_WEIGHTS = {
    "language_complexity": 0.40,
    "term_clarity": 0.30,
    "cultural_references": 0.30
}

# Learning Objectives sub-weights
OBJECTIVES_WEIGHTS = {
    "bloom_taxonomy": 0.40,
    "dok_level": 0.40,
    "contribution_assessment": 0.20
}

# Initialize Anthropic client
def initialize_anthropic_client():
    """Initialize and return the Anthropic API client."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    
    # Validate API key before attempting to create client
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables. Please set this environment variable before running.")
    
    if api_key.startswith("sk-") == False:
        raise ValueError("ANTHROPIC_API_KEY appears to be invalid. It should start with 'sk-'")
        
    try:
        return anthropic.Anthropic(api_key=api_key)
    except Exception as e:
        raise ValueError(f"Failed to initialize Anthropic client: {str(e)}")

def requires_client(func):
    """
    Decorator to standardize client validation across functions.
    Raises an error if client is None.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Check if client is in kwargs
        client = kwargs.get('client')
        # If not in kwargs, check if it's in args (usually the second parameter)
        if client is None and len(args) >= 2:
            client = args[1]
        
        if client is None:
            raise ValueError(f"Function {func.__name__} requires a valid LLM client, but received None")
        
        return func(*args, **kwargs)
    return wrapper

# LLM Interaction Functions
# Apply the decorator to the LLM response function
@cache_llm_response
def generate_llm_response(client, prompt, max_tokens=2000, model="claude-3-7-sonnet-20250219"):
    """
    Generate a response from Claude with improved error handling and caching.
    
    Args:
        client: An initialized Anthropic client
        prompt: The prompt to send to the model
        max_tokens: Maximum tokens to generate in the response
        model: The Claude model to use (default: claude-3-7-sonnet-20250219)
        
    Returns:
        The text response from the model
        
    Raises:
        Exception: If the API call fails
    """
    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.0  # Use deterministic output for consistent evaluations
        )
        
        # Extract text with better error handling
        text_content = ""
        
        # Check if response exists
        if not response:
            raise ValueError("Empty response received from API")
            
        # Check if content attribute exists
        if not hasattr(response, 'content'):
            raise ValueError("Response missing 'content' attribute, API may have changed")
            
        # Check if content is a non-empty list/iterable
        if not response.content or not hasattr(response.content, '__iter__'):
            raise ValueError("Response content is empty or not iterable")
        
        # Process each content block
        for content_block in response.content:
            # Check if text attribute exists in block
            if hasattr(content_block, 'text') and content_block.text:
                text_content += content_block.text
            # If there's a type attribute, try to handle different content types
            elif hasattr(content_block, 'type'):
                if content_block.type == 'text':
                    # Direct text field
                    if hasattr(content_block, 'text'):
                        text_content += content_block.text
                    # Value field (potential API change)
                    elif hasattr(content_block, 'value'):
                        text_content += content_block.value
                # Handle other potential content types if needed
                else:
                    print(f"Skipping unsupported content type: {content_block.type}")
        
        if not text_content:
            print("Warning: Extracted empty text from response, response structure may have changed")
            
        return text_content
            
    except Exception as e:
        print(f"Error generating LLM response: {str(e)}")
        raise

def extract_json_from_response(response):
    """
    Extract JSON data from the LLM response with improved error handling.
    
    Args:
        response: The text response from the LLM
        
    Returns:
        The parsed JSON data as a Python dictionary/list
        
    Raises:
        ValueError: If no valid JSON is found in the response
    """
    # Log the original response for debugging
    print(f"Attempting to extract JSON from response (first 100 chars): {response[:100]}...")
    
    # Try to find JSON content enclosed within triple backticks (markdown code blocks)
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
    
    if json_match:
        # Extract content from the code block
        json_str = json_match.group(1)
        print(f"Found JSON in code block: {json_str[:100]}...")
    else:
        # If no code blocks, look for content that appears to be JSON
        json_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', response)
        if json_match:
            json_str = json_match.group(1)
            print(f"Found JSON without code block: {json_str[:100]}...")
        else:
            print(f"No JSON pattern found in response: {response}")
            return {"score": 0, "explanation": "Could not parse JSON from LLM response"}
    
    # Try multiple JSON parsing strategies
    try:
        # First attempt: direct parsing
        return json.loads(json_str)
    except json.JSONDecodeError as e1:
        print(f"Initial JSON parsing failed: {str(e1)}")
        
        try:
            # Second attempt: Remove non-ASCII chars
            clean_json_str = re.sub(r'[^\x00-\x7F]+', '', json_str)
            return json.loads(clean_json_str)
        except json.JSONDecodeError as e2:
            print(f"ASCII-cleaned JSON parsing failed: {str(e2)}")
            
            try:
                # Third attempt: Fix common escaping issues
                clean_json_str = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', '', clean_json_str)
                return json.loads(clean_json_str)
            except json.JSONDecodeError as e3:
                print(f"Escape-fixed JSON parsing failed: {str(e3)}")
                
                try:
                    # Fourth attempt: Handle trailing commas (common LLM error)
                    clean_json_str = re.sub(r',\s*([}\]])', r'\1', clean_json_str)
                    return json.loads(clean_json_str)
                except json.JSONDecodeError as e4:
                    print(f"Comma-fixed JSON parsing failed: {str(e4)}")
                    
                    try:
                        # Fifth attempt: Fix unquoted keys (another common LLM error)
                        clean_json_str = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', clean_json_str)
                        return json.loads(clean_json_str)
                    except json.JSONDecodeError as e5:
                        print(f"All JSON parsing attempts failed. Original response:\n{response}\n\nExtracted JSON:\n{json_str}")
                        # Return a default response
                        return {"score": 0, "explanation": "Could not parse JSON from LLM response"}

# Data Loading Functions
def load_questions_from_json(file_path):
    """
    Load questions from a JSON file.
    
    Args:
        file_path: Path to the JSON file containing questions
        
    Returns:
        A list of question dictionaries
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
        ValueError: If the file doesn't contain properly formatted question data
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Open and parse the JSON file
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Handle different possible JSON structures
        questions = []
        
        # Case 1: Data is a list of questions directly
        if isinstance(data, list):
            questions = data
        # Case 2: Data has a 'questions' key containing the list
        elif isinstance(data, dict) and 'questions' in data:
            questions = data['questions']
        # Case 3: Data has some other structure we need to extract from
        else:
            # Try to find any list that might contain question data
            for key, value in data.items():
                if isinstance(value, list) and len(value) > 0:
                    # Check if the first item looks like a question
                    if isinstance(value[0], dict) and 'question' in value[0]:
                        questions = value
                        break
        
        # Validate that we found some questions
        if not questions:
            raise ValueError("No questions found in the JSON file")
        
        # Validate the structure of each question
        for i, question in enumerate(questions):
            if not isinstance(question, dict):
                raise ValueError(f"Question {i} is not a dictionary: {question}")
            
            if 'question' not in question:
                raise ValueError(f"Question {i} is missing the 'question' field: {question}")
        
        print(f"Successfully loaded {len(questions)} questions from {file_path}")
        return questions
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON from {file_path}: {str(e)}")
        raise
    except Exception as e:
        print(f"Error loading questions from {file_path}: {str(e)}")
        raise

def validate_question_data(question_data):
    """
    Validate the question data structure and ensure required fields are present.
    
    Args:
        question_data: Dictionary containing the question data
        
    Returns:
        bool: True if valid, False otherwise
        str: Error message if invalid, None otherwise
    """
    # Check if question_data is a dictionary
    if not isinstance(question_data, dict):
        return False, f"Question data must be a dictionary, got {type(question_data).__name__}"
    
    # Check required fields
    if "question" not in question_data:
        return False, "Missing required field: 'question'"
    
    # Check question is a non-empty string
    if not isinstance(question_data["question"], str) or not question_data["question"].strip():
        return False, "Field 'question' must be a non-empty string"
    
    # Check optional fields have correct types if present
    if "passage" in question_data and not isinstance(question_data["passage"], str):
        return False, "Field 'passage' must be a string"
    
    if "options" in question_data:
        if not isinstance(question_data["options"], (list, dict)):
            return False, "Field 'options' must be a list or dictionary"
        if isinstance(question_data["options"], list) and not all(isinstance(opt, str) for opt in question_data["options"]):
            return False, "All options in 'options' list must be strings"
    
    if "target_grade" in question_data and not isinstance(question_data["target_grade"], (int, float)):
        return False, "Field 'target_grade' must be a number"
    
    return True, None

# Dimension 1: Knowledge Requirements (30%)
@requires_client
def aggregate_knowledge_requirements(question_data, client=None):
    """
    Function to combine and calculate the metrics for each element in this dimension.
    Takes in the scores of each of the relevant checks in this dimension and combines them based on weight into a dimension score.
    30% of overall score.
    """
    question_text = question_data["question"]
    passage_text = question_data.get("passage", "")
    
    # Get scores for each component (placeholders)
    prior_knowledge_score, prior_knowledge_details = evaluate_prior_knowledge(question_data, client)
    domain_specific_score, domain_specific_details = evaluate_domain_specific(question_data, client)
    knowledge_level_score, knowledge_level_details = evaluate_knowledge_level(question_data, client)
    concept_abstractness_score, concept_abstractness_details = evaluate_concept_abstractness(question_data, client)
    specialized_terminology_score, specialized_terminology_details = evaluate_specialized_terminology(question_text)
    
    # Store all component results
    component_results = {
        "prior_knowledge": {
            "score": prior_knowledge_score,
            "weight": KNOWLEDGE_WEIGHTS["prior_knowledge"],
            "details": prior_knowledge_details
        },
        "domain_specific": {
            "score": domain_specific_score,
            "weight": KNOWLEDGE_WEIGHTS["domain_specific"],
            "details": domain_specific_details
        },
        "knowledge_level": {
            "score": knowledge_level_score,
            "weight": KNOWLEDGE_WEIGHTS["knowledge_level"],
            "details": knowledge_level_details
        },
        "concept_abstractness": {
            "score": concept_abstractness_score,
            "weight": KNOWLEDGE_WEIGHTS["concept_abstractness"],
            "details": concept_abstractness_details
        },
        "specialized_terminology": {
            "score": specialized_terminology_score,
            "weight": KNOWLEDGE_WEIGHTS["specialized_terminology"],
            "details": specialized_terminology_details
        }
    }
    
    # Calculate weighted score
    weighted_score = calculate_dimension_score(component_results)
    
    return weighted_score, component_results

@requires_client
def evaluate_prior_knowledge(question_data, client=None):
    """
    Evaluate how much prior knowledge is required to understand the question.
    Uses an LLM to assess if prior knowledge needed is easy (0) or hard (1).
    
    Args:
        question_data: Dictionary containing the question and related information
        client: Anthropic client instance (optional, can be None for unit testing)
        
    Returns:
        int: Binary difficulty score: 0 (easy) or 1 (hard)
        dict: Detailed assessment information
        
    Note:
        This function contributes 25% to the knowledge requirements dimension score.
    """
    # Return placeholder values if client is None (for testing)
    if client is None:
        return 0, {"explanation": "Client not provided, returning default score"}
    
    question_text = question_data["question"]
    passage_text = question_data.get("passage", "")
    options = question_data.get("options", [])
    
    # Format options string if present
    options_text = ""
    if options:
        if isinstance(options, list):
            options_text = "\nOptions:\n" + "\n".join([f"- {opt}" for opt in options])
        elif isinstance(options, dict):
            options_text = "\nOptions:\n" + "\n".join([f"- {k}: {v}" for k, v in options.items()])
    
    # Construct a detailed prompt for the LLM
    prompt = f"""
You are evaluating the level of prior knowledge required to understand and answer a question.
Your task is to determine if this question requires LITTLE or MUCH prior knowledge.

Make a binary assessment: 0 for LITTLE, 1 for MUCH.

Use these criteria for your assessment:

LITTLE PRIOR KNOWLEDGE (Score 0):
- Minimal prior knowledge needed
- Accessible to beginners in the subject
- Knowledge can be derived from common education or general awareness
- Basic concepts that most people encounter in everyday life or early education

MUCH PRIOR KNOWLEDGE (Score 1):
- Extensive prior knowledge required
- Assumes significant background in the subject area
- Requires understanding of advanced concepts within the field
- Knowledge typically gained through specialized or advanced study

Please analyze the following question:

QUESTION: {question_text}
{f"PASSAGE: {passage_text}" if passage_text else ""}
{options_text}

Provide your assessment in JSON format with the following structure:
```json
{{
  "score": 0 or 1,
  "reasoning": "Your step-by-step analysis of the prior knowledge required",
  "knowledge_areas": ["List of specific knowledge areas required"],
  "explanation": "Summary of why you assigned this score"
}}
```
Focus your evaluation specifically on the prior knowledge requirement, not other aspects of difficulty.
"""
    try:
        # Get response from the LLM
        response = generate_llm_response(client, prompt)

        # Extract and parse the JSON response
        result = extract_json_from_response(response)

        # Validate the result has all required fields
        if not all(k in result for k in ["score", "explanation"]):
            raise ValueError("LLM response missing required fields")
        
        # Ensure the score is binary (0 or 1)
        score = 1 if result["score"] > 0.5 else 0
        if result["score"] not in [0, 1]:
            result["original_score"] = result["score"]
            result["score"] = score
            result["explanation"] += " (Score converted to binary value)"

        return score, result
    
    except Exception as e:
        print(f"Error in evaluate_prior_knowledge: {str(e)}")
        # Return a default score in case of error
        return 0, {
            "score": 0,
            "explanation": f"Error in evaluation: {str(e)}",
            "error": True
        }

@requires_client
def evaluate_domain_specific(question_data, client=None):
    """
    Evaluate if the knowledge required is domain-specific or general.
    Uses an LLM to determine if specialized domain knowledge is needed (hard) or if general knowledge is sufficient (easy).
    
    Args:
        question_text: The text of the question to evaluate
        client: Anthropic client instance (optional, can be None for unit testing)
        
    Returns:
        int: Binary difficulty score: 0 (easy/general knowledge) or 1 (hard/domain-specific)
        dict: Detailed assessment information
        
    Note:
        This function contributes 20% to the knowledge requirements dimension score.
    """
    # Return placeholder values if client is None (for testing)
    if client is None:
        return 0, {"explanation": "Client not provided, returning default score"}

    question_text = question_data["question"]
    passage_text = question_data.get("passage", "")
    options = question_data.get("options", [])
    
    # Format options string if present
    options_text = ""
    if options:
        if isinstance(options, list):
            options_text = "\nOptions:\n" + "\n".join([f"- {opt}" for opt in options])
        elif isinstance(options, dict):
            options_text = "\nOptions:\n" + "\n".join([f"- {k}: {v}" for k, v in options.items()])
    
    # Construct prompt for the LLM
    prompt = f"""
You are evaluating whether a question requires domain-specific knowledge or general knowledge.
Your task is to determine if the question can be answered using general knowledge (EASY) or if it requires specialized domain knowledge (HARD).

Make a binary assessment: 0 for EASY (general knowledge), 1 for HARD (domain-specific).

Use these criteria for your assessment:

EASY (Score 0):
- General knowledge that most educated people possess
- Common concepts taught in basic education
- Knowledge that appears frequently in everyday life or popular media
- Information that can be reasonably expected across different backgrounds

HARD (Score 1):
- Highly specialized domain-specific knowledge
- Technical concepts limited to specific fields of study
- Knowledge typically gained through specialized education or training
- Terminology or concepts that would be unfamiliar to those outside the field

Please analyze the following question:

QUESTION: {question_text}
{f"PASSAGE: {passage_text}" if passage_text else ""}
{options_text}

Provide your assessment in JSON format with the following structure:
```json
{{
  "score": 0 or 1,
  "domain": "The specific domain/field of knowledge required (if applicable)",
  "reasoning": "Your step-by-step analysis of the domain-specificity",
  "explanation": "Summary of why you assigned this score"
}}
```
Focus your evaluation specifically on whether the knowledge is general or domain-specific, not other aspects of difficulty.
"""
    try:
        # Get response from the LLM
        response = generate_llm_response(client, prompt)
        
        # Extract and parse the JSON response
        result = extract_json_from_response(response)
        
        # Validate the result has required fields
        if not all(k in result for k in ["score", "explanation"]):
            raise ValueError("LLM response missing required fields")
        
        # Ensure the score is binary (0 or 1)
        score = 1 if result["score"] > 0.5 else 0
        if result["score"] not in [0, 1]:
            result["original_score"] = result["score"]
            result["score"] = score
            result["explanation"] += " (Score converted to binary value)"
        
        return score, result

    except Exception as e:
        print(f"Error in evaluate_domain_specific: {str(e)}")
        # Return a default score in case of error
        return 0, {
            "score": 0,
            "explanation": f"Error in evaluation: {str(e)}",
            "error": True
    }

@requires_client
def evaluate_knowledge_level(question_data, client=None):
    """
    Evaluate what level of subject knowledge is required to answer the question.
    Uses an LLM to determine if basic knowledge is sufficient (easy) or if advanced knowledge is required (hard).
    
    Args:
        question_text: The text of the question to evaluate
        client: Anthropic client instance (optional, can be None for unit testing)
        
    Returns:
        int: Binary difficulty score: 0 (easy/basic knowledge) or 1 (hard/advanced knowledge)
        dict: Detailed assessment information
        
    Note:
        This function contributes 25% to the knowledge requirements dimension score.
    """
    # Return placeholder values if client is None (for testing)
    if client is None:
        return 0, {"explanation": "Client not provided, returning default score"}

    question_text = question_data["question"]
    passage_text = question_data.get("passage", "")
    options = question_data.get("options", [])
    
    # Format options string if present
    options_text = ""
    if options:
        if isinstance(options, list):
            options_text = "\nOptions:\n" + "\n".join([f"- {opt}" for opt in options])
        elif isinstance(options, dict):
            options_text = "\nOptions:\n" + "\n".join([f"- {k}: {v}" for k, v in options.items()])
    
    # Construct prompt for the LLM
    prompt = f"""
You are evaluating the level of subject knowledge required to answer a question.
Your task is to determine if the question requires BASIC knowledge (EASY) or ADVANCED knowledge (HARD).

Make a binary assessment: 0 for EASY (basic knowledge), 1 for HARD (advanced knowledge).

Use these criteria for your assessment:

EASY (Score 0):
- Basic, introductory level knowledge
- Concepts covered in early stages of learning the subject
- Foundational principles that are typically taught first
- Knowledge expected after minimal exposure to the subject
- Commonly addressed in introductory courses or materials

HARD (Score 1):
- Advanced, expert-level knowledge
- Concepts typically covered in later stages of learning
- In-depth understanding of the subject matter
- Knowledge that builds upon multiple foundational concepts
- Content usually found in advanced courses or specialized materials
- Requires synthesis of multiple concepts or nuanced understanding

Please analyze the following question:

QUESTION: {question_text}
{f"PASSAGE: {passage_text}" if passage_text else ""}
{options_text}

Provide your assessment in JSON format with the following structure:
```json
{{
  "score": 0 or 1,
  "subject_area": "The specific subject area of the question",
  "knowledge_level": "Your assessment of the knowledge level (basic, intermediate, or advanced)",
  "reasoning": "Your step-by-step analysis of the knowledge level required",
  "explanation": "Summary of why you assigned this score"
}}
```
Focus your evaluation specifically on the level of subject knowledge required, not other aspects of difficulty.
"""
    try:
        # Get response from the LLM
        response = generate_llm_response(client, prompt)
        
        # Extract and parse the JSON response
        result = extract_json_from_response(response)
        
        # Validate the result has required fields
        if not all(k in result for k in ["score", "explanation"]):
            raise ValueError("LLM response missing required fields")
        
        # Ensure the score is binary (0 or 1)
        score = 1 if result["score"] > 0.5 else 0
        if result["score"] not in [0, 1]:
            result["original_score"] = result["score"]
            result["score"] = score
            result["explanation"] += " (Score converted to binary value)"
        
        return score, result

    except Exception as e:
        print(f"Error in evaluate_knowledge_level: {str(e)}")
        # Return a default score in case of error
        return 0, {
            "score": 0,
            "explanation": f"Error in evaluation: {str(e)}",
            "error": True
        }

@requires_client
def evaluate_concept_abstractness(question_data, client=None):
    """
    Evaluate if the concepts involved in the question are abstract/complex or concrete/straightforward.
    Uses an LLM to determine if concepts are concrete (easy) or abstract (hard).
    
    Args:
        question_text: The text of the question to evaluate
        client: Anthropic client instance (optional, can be None for unit testing)
        
    Returns:
        int: Binary difficulty score: 0 (easy/concrete) or 1 (hard/abstract)
        dict: Detailed assessment information
        
    Note:
        This function contributes 15% to the knowledge requirements dimension score.
    """
    # Return placeholder values if client is None (for testing)
    if client is None:
        return 0, {"explanation": "Client not provided, returning default score"}

    question_text = question_data["question"]
    passage_text = question_data.get("passage", "")
    options = question_data.get("options", [])
    
    # Format options string if present
    options_text = ""
    if options:
        if isinstance(options, list):
            options_text = "\nOptions:\n" + "\n".join([f"- {opt}" for opt in options])
        elif isinstance(options, dict):
            options_text = "\nOptions:\n" + "\n".join([f"- {k}: {v}" for k, v in options.items()])
    
    # Construct prompt for the LLM
    prompt = f"""
You are evaluating the level of abstractness and complexity of concepts in a question.
Your task is to determine if the question involves CONCRETE, STRAIGHTFORWARD concepts (EASY) or ABSTRACT, COMPLEX concepts (HARD).

Make a binary assessment: 0 for EASY (concrete/straightforward), 1 for HARD (abstract/complex).

Use these criteria for your assessment:

EASY (Score 0):
- Concrete, straightforward concepts
- Directly observable or measurable phenomena
- Literal rather than figurative understanding
- Clear, definite relationships between concepts
- Concepts that can be easily visualized or demonstrated
- Basic relationships without many layers of abstraction

HARD (Score 1):
- Highly abstract or complex theoretical concepts
- Requires understanding of intangible or conceptual ideas
- Involves multiple layers of abstraction
- Concepts that are difficult to visualize directly
- Requires understanding complex interactions between multiple variables
- Involves metaphorical thinking or representation

Please analyze the following question:

QUESTION: {question_text}
{f"PASSAGE: {passage_text}" if passage_text else ""}
{options_text}

Provide your assessment in JSON format with the following structure:
```json
{{
  "score": 0 or 1,
  "concepts_identified": ["List of key concepts in the question"],
  "abstractness_level": "Your assessment of the abstractness level",
  "reasoning": "Your step-by-step analysis of the abstractness and complexity",
  "explanation": "Summary of why you assigned this score"
}}
```
Focus your evaluation specifically on the abstractness and complexity of the concepts involved, not other aspects of difficulty.
"""
    try:
        # Get response from the LLM
        response = generate_llm_response(client, prompt)
        
        # Extract and parse the JSON response
        result = extract_json_from_response(response)
        
        # Validate the result has required fields
        if not all(k in result for k in ["score", "explanation"]):
            raise ValueError("LLM response missing required fields")
        
        # Ensure the score is binary (0 or 1)
        score = 1 if result["score"] > 0.5 else 0
        if result["score"] not in [0, 1]:
            result["original_score"] = result["score"]
            result["score"] = score
            result["explanation"] += " (Score converted to binary value)"
        
        return score, result

    except Exception as e:
        print(f"Error in evaluate_concept_abstractness: {str(e)}")
        # Return a default score in case of error
        return 0, {
            "score": 0,
            "explanation": f"Error in evaluation: {str(e)}",
            "error": True
        }

def evaluate_specialized_terminology(question_text, client=None):
    """
    Evaluate if the question uses specialized terminology that might be unfamiliar to some learners.
    
    Args:
        question_text: The text of the question to evaluate
        client: Anthropic client instance (not used in this function)
        
    Returns:
        int: Binary difficulty score: 0 (easy/common terms) or 1 (hard/specialized terms)
        dict: Detailed assessment information
        
    Note:
        This function contributes 15% to the knowledge requirements dimension score.
    """
    try:
        # Path to the academic terms file - adjust as needed
        academic_terms_file = os.path.join(os.path.dirname(__file__), "data", "academic_terms.txt")
        
        # Check if the file exists
        if not os.path.exists(academic_terms_file):
            raise FileNotFoundError(f"Academic terms file not found at: {academic_terms_file}")
        
        # Load academic terms from file
        with open(academic_terms_file, 'r', encoding='utf-8') as file:
            academic_terms = {line.strip() for line in file if line.strip()}
        
        # Preprocess the question text
        question_text = question_text.lower()
        # Remove punctuation and split into words
        words = re.sub(r'[^\w\s]', ' ', question_text).split()
        
        # Find single-word academic terms in the question
        academic_words_found = [word for word in words if word in academic_terms]
        
        # Find multi-word academic terms in the question
        multi_word_terms_found = []
        for i in range(len(words)):
            for j in range(2, 6):  # Check phrases of 2-5 words
                if i + j <= len(words):
                    phrase = ' '.join(words[i:i+j])
                    if phrase in academic_terms:
                        multi_word_terms_found.append(phrase)
        
        # Calculate metrics
        all_terms_found = set(academic_words_found + multi_word_terms_found)
        total_words = len(words)
        
        # Calculate ratios
        term_ratio = len(all_terms_found) / max(total_words, 1)
        unique_term_ratio = len(all_terms_found) / max(len(set(words)), 1)
        
        # Define thresholds for hard vs. easy classification
        TERM_RATIO_THRESHOLD = 0.25  # If >25% of words are academic terms
        TERM_COUNT_THRESHOLD = 4     # If ≥4 unique academic terms
        
        # Determine difficulty score (0 for easy, 1 for hard)
        is_hard = (term_ratio >= TERM_RATIO_THRESHOLD) or (len(all_terms_found) >= TERM_COUNT_THRESHOLD)
        difficulty_score = 1 if is_hard else 0
        
        # Return score and details
        return difficulty_score, {
            "total_words": total_words,
            "academic_terms_count": len(all_terms_found),
            "academic_terms_found": list(all_terms_found),
            "term_ratio": term_ratio,
            "unique_term_ratio": unique_term_ratio,
            "is_hard": is_hard
        }
    
    except Exception as e:
        print(f"Error in evaluate_specialized_terminology: {str(e)}")
        return 0, {
            "error": True,
            "explanation": f"Error evaluating specialized terminology: {str(e)}"
        }

# Dimension 2: Cognitive Complexity (30%)
@requires_client
def aggregate_cognitive_complexity(question_data, client=None):
    """
    Function to combine and calculate the metrics for each element in this dimension.
    Takes in the scores of each of the relevant checks in this dimension and combines them based on weight into a dimension score.
    30% of overall score.
    """
    question_text = question_data["question"]
    
    # Get scores for each component (placeholders)
    thinking_score, thinking_details = evaluate_thinking_requirements(question_data, client)
    cognitive_level_score, cognitive_level_details = estimate_cognitive_level(question_data, client)
    problem_complexity_score, problem_complexity_details = evaluate_problem_complexity(question_data, client)
    cognitive_bias_score, cognitive_bias_details = evaluate_cognitive_bias(question_data, client)
    
    # Store all component results
    component_results = {
        "thinking_requirements": {
            "score": thinking_score,
            "weight": COGNITIVE_WEIGHTS["thinking_requirements"],
            "details": thinking_details
        },
        "cognitive_level": {
            "score": cognitive_level_score,
            "weight": COGNITIVE_WEIGHTS["cognitive_level"],
            "details": cognitive_level_details
        },
        "problem_complexity": {
            "score": problem_complexity_score,
            "weight": COGNITIVE_WEIGHTS["problem_complexity"],
            "details": problem_complexity_details
        },
        "cognitive_bias": {
            "score": cognitive_bias_score,
            "weight": COGNITIVE_WEIGHTS["cognitive_bias"],
            "details": cognitive_bias_details
        }
    }
    
    # Calculate weighted score
    weighted_score = calculate_dimension_score(component_results)
    
    return weighted_score, component_results

@requires_client
def evaluate_thinking_requirements(question_data, client=None):
    """
    Evaluate if the question requires critical thinking, problem-solving, or other higher-order thinking skills.
    Uses an LLM to determine if basic thinking skills suffice (easy) or if higher-order thinking is required (hard).
    
    Args:
        question_data: Dictionary containing the question and related information
        client: Anthropic client instance (optional, can be None for unit testing)
        
    Returns:
        int: Binary difficulty score: 0 (easy/basic thinking) or 1 (hard/higher-order thinking)
        dict: Detailed assessment information
        
    Note:
        This function contributes 30% to the cognitive complexity dimension score.
    """
    # Return placeholder values if client is None (for testing)
    if client is None:
        return 0, {"explanation": "Client not provided, returning default score"}
    
    question_text = question_data["question"]
    passage_text = question_data.get("passage", "")
    options = question_data.get("options", [])
    
    # Format options string if present
    options_text = ""
    if options:
        if isinstance(options, list):
            options_text = "\nOptions:\n" + "\n".join([f"- {opt}" for opt in options])
        elif isinstance(options, dict):
            options_text = "\nOptions:\n" + "\n".join([f"- {k}: {v}" for k, v in options.items()])
    
    # Construct a detailed prompt for the LLM
    prompt = f"""
You are evaluating the cognitive thinking requirements of a question.
Your task is to determine if the question requires BASIC THINKING SKILLS (EASY) or HIGHER-ORDER THINKING SKILLS (HARD).

Make a binary assessment: 0 for EASY (basic thinking), 1 for HARD (higher-order thinking).

Use these criteria for your assessment:

EASY (Score 0):
- Straightforward application of knowledge
- Simple recall or recognition of information
- Direct comprehension without deep analysis
- Following clear procedures without adaptation
- Single-step mental processes
- Clear, direct path to the answer

HARD (Score 1):
- Requires critical thinking and analysis
- Involves evaluating, synthesizing, or creating
- Problem-solving requiring multiple steps or approaches
- Applying concepts to novel situations
- Requires making judgments or decisions based on criteria
- Drawing connections between multiple concepts
- Identifying patterns or relationships not explicitly stated

Please analyze the following question:

QUESTION: {question_text}
{f"PASSAGE: {passage_text}" if passage_text else ""}
{options_text}

Provide your assessment in JSON format with the following structure:
```json
{{
  "score": 0 or 1,
  "thinking_skills_required": ["List of specific thinking skills required"],
  "cognitive_level": "Your assessment of the cognitive level (recall, understand, apply, analyze, evaluate, create)",
  "reasoning": "Your step-by-step analysis of the thinking requirements",
  "explanation": "Summary of why you assigned this score"
}}
```
Focus your evaluation specifically on the thinking skills required, not other aspects of difficulty.
"""
    try:
        # Get response from the LLM
        response = generate_llm_response(client, prompt)
        
        # Extract and parse the JSON response
        result = extract_json_from_response(response)
        
        # Validate the result has required fields
        if not all(k in result for k in ["score", "explanation"]):
            raise ValueError("LLM response missing required fields")
        
        # Ensure the score is binary (0 or 1)
        score = 1 if result["score"] > 0.5 else 0
        if result["score"] not in [0, 1]:
            result["original_score"] = result["score"]
            result["score"] = score
            result["explanation"] += " (Score converted to binary value)"
        
        return score, result

    except Exception as e:
        print(f"Error in evaluate_thinking_requirements: {str(e)}")
        # Return a default score in case of error
        return 0, {
            "score": 0,
            "explanation": f"Error in evaluation: {str(e)}",
            "error": True
        }

@requires_client
def estimate_cognitive_level(question_data, client=None):
    """
    Estimate what cognitive process is required to answer the question.
    Uses an LLM to determine if lower-order cognitive processes (recall, comprehension) suffice (easy)
    or if higher-order processes (application, analysis, synthesis, evaluation) are required (hard).
    
    Args:
        question_text: The text of the question to evaluate
        client: Anthropic client instance (optional, can be None for unit testing)
        
    Returns:
        int: Binary difficulty score: 0 (easy/lower-order) or 1 (hard/higher-order)
        dict: Detailed assessment information
        
    Note:
        This function contributes 30% to the cognitive complexity dimension score.
    """
    # Return placeholder values if client is None (for testing)
    if client is None:
        return 0, {"explanation": "Client not provided, returning default score"}

    question_text = question_data["question"]
    passage_text = question_data.get("passage", "")
    options = question_data.get("options", [])
    
    # Format options string if present
    options_text = ""
    if options:
        if isinstance(options, list):
            options_text = "\nOptions:\n" + "\n".join([f"- {opt}" for opt in options])
        elif isinstance(options, dict):
            options_text = "\nOptions:\n" + "\n".join([f"- {k}: {v}" for k, v in options.items()])
    
    # Construct prompt for the LLM
    prompt = f"""
You are evaluating the cognitive process level required to answer a question based on Bloom's Taxonomy.
Your task is to determine if the question requires LOWER-ORDER THINKING (EASY) or HIGHER-ORDER THINKING (HARD).

Make a binary assessment: 0 for EASY (lower-order), 1 for HARD (higher-order).

Use these criteria from Bloom's Taxonomy:

EASY (Score 0) - Lower-order thinking:
- Remember: Recall facts and basic concepts
  (Keywords: define, list, memorize, repeat, state, name, identify)
- Understand: Explain ideas or concepts
  (Keywords: describe, discuss, explain, locate, recognize, report, summarize)

HARD (Score 1) - Higher-order thinking:
- Apply: Use information in new situations
  (Keywords: apply, demonstrate, illustrate, operate, solve, use, calculate)
- Analyze: Draw connections among ideas
  (Keywords: analyze, categorize, compare, contrast, distinguish, examine, test)
- Evaluate: Justify a stand or decision
  (Keywords: appraise, argue, defend, judge, critique, support, value, evaluate)
- Create: Produce new or original work
  (Keywords: design, develop, formulate, construct, create, compose, generate)

Please analyze the following question:

QUESTION: {question_text}
{f"PASSAGE: {passage_text}" if passage_text else ""}
{options_text}

Provide your assessment in JSON format with the following structure:
```json
{{
  "score": 0 or 1,
  "primary_level": "The main cognitive level required (remember, understand, apply, analyze, evaluate, create)",
  "bloom_verbs_identified": ["Specific Bloom's taxonomy verbs identified in the question"],
  "reasoning": "Your step-by-step analysis of the cognitive process required",
  "explanation": "Summary of why you assigned this score"
}}
```
Focus your evaluation specifically on the cognitive process level required, not other aspects of difficulty.
"""
    try:
        # Get response from the LLM
        response = generate_llm_response(client, prompt)
        
        # Extract and parse the JSON response
        result = extract_json_from_response(response)
        
        # Validate the result has required fields
        if not all(k in result for k in ["score", "explanation"]):
            raise ValueError("LLM response missing required fields")
        
        # Ensure the score is binary (0 or 1)
        score = 1 if result["score"] > 0.5 else 0
        if result["score"] not in [0, 1]:
            result["original_score"] = result["score"]
            result["score"] = score
            result["explanation"] += " (Score converted to binary value)"
        
        return score, result

    except Exception as e:
        print(f"Error in estimate_cognitive_level: {str(e)}")
        # Return a default score in case of error
        return 0, {
            "score": 0,
            "explanation": f"Error in evaluation: {str(e)}",
            "error": True
        }

@requires_client
def evaluate_problem_complexity(question_data, client=None):
    """
    Evaluate if the question involves a straightforward problem or a complex, open-ended one.
    Uses an LLM to determine if the problem is simple and singular (easy) or complex and multi-faceted (hard).
    
    Args:
        question_text: The text of the question to evaluate
        client: Anthropic client instance (optional, can be None for unit testing)
        
    Returns:
        int: Binary difficulty score: 0 (easy/straightforward) or 1 (hard/complex)
        dict: Detailed assessment information
        
    Note:
        This function contributes 20% to the cognitive complexity dimension score.
    """
    # Return placeholder values if client is None (for testing)
    if client is None:
        return 0, {"explanation": "Client not provided, returning default score"}

    question_text = question_data["question"]
    passage_text = question_data.get("passage", "")
    options = question_data.get("options", [])
    
    # Format options string if present
    options_text = ""
    if options:
        if isinstance(options, list):
            options_text = "\nOptions:\n" + "\n".join([f"- {opt}" for opt in options])
        elif isinstance(options, dict):
            options_text = "\nOptions:\n" + "\n".join([f"- {k}: {v}" for k, v in options.items()])
    
    # Construct prompt for the LLM
    prompt = f"""
You are evaluating the complexity of a problem presented in a question.
Your task is to determine if the question involves a SIMPLE, STRAIGHTFORWARD PROBLEM (EASY) or a COMPLEX, MULTI-STEP PROBLEM (HARD).

Make a binary assessment: 0 for EASY (straightforward), 1 for HARD (complex).

Use these criteria for your assessment:

EASY (Score 0):
- Simple, single-step problem
- Clear, well-defined path to solution
- Only requires consideration of one or two factors
- Limited number of variables or constraints
- Routine problem with standard solution method
- Closed-ended with a single correct answer

HARD (Score 1):
- Complex, multi-step problem with various considerations
- Requires developing or selecting a problem-solving strategy
- Multiple factors, variables, or constraints to consider
- May have multiple valid approaches or solutions
- Requires integration of different concepts or procedures
- May be open-ended or ill-structured
- Could involve ambiguity or uncertainty

Please analyze the following question:

QUESTION: {question_text}
{f"PASSAGE: {passage_text}" if passage_text else ""}
{options_text}

Provide your assessment in JSON format with the following structure:
```json
{{
  "score": 0 or 1,
  "steps_required": "Estimated number of steps to solve the problem",
  "problem_type": "Classification of the problem (e.g., well-defined, ill-defined, routine, non-routine)",
  "factors_to_consider": ["List of factors or variables that need to be considered"],
  "reasoning": "Your step-by-step analysis of the problem complexity",
  "explanation": "Summary of why you assigned this score"
}}
```
Focus your evaluation specifically on the problem complexity, not other aspects of difficulty.
"""
    try:
        # Get response from the LLM
        response = generate_llm_response(client, prompt)
        
        # Extract and parse the JSON response
        result = extract_json_from_response(response)
        
        # Validate the result has required fields
        if not all(k in result for k in ["score", "explanation"]):
            raise ValueError("LLM response missing required fields")
        
        # Ensure the score is binary (0 or 1)
        score = 1 if result["score"] > 0.5 else 0
        if result["score"] not in [0, 1]:
            result["original_score"] = result["score"]
            result["score"] = score
            result["explanation"] += " (Score converted to binary value)"
        
        return score, result

    except Exception as e:
        print(f"Error in evaluate_problem_complexity: {str(e)}")
        # Return a default score in case of error
        return 0, {
            "score": 0,
            "explanation": f"Error in evaluation: {str(e)}",
            "error": True
        }

@requires_client
def evaluate_cognitive_bias(question_data, client=None):
    """
    Evaluate if there are cognitive biases or misconceptions that could affect the test-taker's response.
    Uses an LLM to determine if the question is relatively free from bias triggers (easy) or likely to 
    activate cognitive biases (hard).
    
    Args:
        question_text: The text of the question to evaluate
        client: Anthropic client instance (optional, can be None for unit testing)
        
    Returns:
        int: Binary difficulty score: 0 (easy/minimal bias potential) or 1 (hard/significant bias potential)
        dict: Detailed assessment information
        
    Note:
        This function contributes 20% to the cognitive complexity dimension score.
    """
    # Return placeholder values if client is None (for testing)
    if client is None:
        return 0, {"explanation": "Client not provided, returning default score"}

    question_text = question_data["question"]
    passage_text = question_data.get("passage", "")
    options = question_data.get("options", [])
    
    # Format options string if present
    options_text = ""
    if options:
        if isinstance(options, list):
            options_text = "\nOptions:\n" + "\n".join([f"- {opt}" for opt in options])
        elif isinstance(options, dict):
            options_text = "\nOptions:\n" + "\n".join([f"- {k}: {v}" for k, v in options.items()])
    
    # Construct prompt for the LLM
    prompt = f"""
You are evaluating whether a question is likely to trigger cognitive biases or misconceptions.
Your task is to determine if the question has MINIMAL BIAS POTENTIAL (EASY) or SIGNIFICANT BIAS POTENTIAL (HARD).

Make a binary assessment: 0 for EASY (minimal bias potential), 1 for HARD (significant bias potential).

Use these criteria for your assessment:

EASY (Score 0):
- Minimal impact of cognitive biases on answering
- Straightforward factual content with little room for misinterpretation
- Question avoids common misconception areas
- Minimal interference from intuitive but incorrect thinking
- Clear wording that doesn't lead toward common errors

HARD (Score 1):
- Strong potential for cognitive biases to interfere
- Involves topics with common misconceptions
- May trigger intuitive but incorrect responses
- Contains elements that might activate anchoring, framing, or availability biases
- Structure or content could lead to representativeness heuristic or confirmation bias
- Requires overcoming preconceived notions or mental shortcuts

Please analyze the following question:

QUESTION: {question_text}
{f"PASSAGE: {passage_text}" if passage_text else ""}
{options_text}

Provide your assessment in JSON format with the following structure:
```json
{{
  "score": 0 or 1,
  "potential_biases": ["List of specific cognitive biases or misconceptions that might be triggered"],
  "bias_triggers": ["Elements in the question that might trigger these biases"],
  "reasoning": "Your step-by-step analysis of the bias potential",
  "explanation": "Summary of why you assigned this score"
}}
```
Focus your evaluation specifically on cognitive bias potential, not other aspects of difficulty.
"""
    try:
        # Get response from the LLM
        response = generate_llm_response(client, prompt)
        
        # Extract and parse the JSON response
        result = extract_json_from_response(response)
        
        # Validate the result has required fields
        if not all(k in result for k in ["score", "explanation"]):
            raise ValueError("LLM response missing required fields")
        
        # Ensure the score is binary (0 or 1)
        score = 1 if result["score"] > 0.5 else 0
        if result["score"] not in [0, 1]:
            result["original_score"] = result["score"]
            result["score"] = score
            result["explanation"] += " (Score converted to binary value)"
        
        return score, result

    except Exception as e:
        print(f"Error in evaluate_cognitive_bias: {str(e)}")
        # Return a default score in case of error
        return 0, {
            "score": 0,
            "explanation": f"Error in evaluation: {str(e)}",
            "error": True
        }

# Dimension 3: Context and Ambiguity (15%)
@requires_client
def aggregate_context_and_ambiguity(question_data, client=None):
    """
    Function to combine and calculate the metrics for each element in this dimension.
    Takes in the scores of each of the relevant checks in this dimension and combines them based on weight into a dimension score.
    15% of overall score
    """
    question_text = question_data["question"]
    passage_text = question_data.get("passage", "")
    
    # Get scores for each component (placeholders)
    context_dependency_score, context_dependency_details = assess_context_dependency(question_text, passage_text, client)
    context_clarity_score, context_clarity_details = assess_context_clarity(question_text, passage_text, client)
    context_difficulty_score, context_difficulty_details = assess_context_difficulty(question_text, passage_text, client)
    
    # Store all component results
    component_results = {
        "context_dependency": {
            "score": context_dependency_score,
            "weight": CONTEXT_WEIGHTS["context_dependency"],
            "details": context_dependency_details
        },
        "context_clarity": {
            "score": context_clarity_score,
            "weight": CONTEXT_WEIGHTS["context_clarity"],
            "details": context_clarity_details
        },
        "context_difficulty": {
            "score": context_difficulty_score,
            "weight": CONTEXT_WEIGHTS["context_difficulty"],
            "details": context_difficulty_details
        }
    }
    
    # Calculate weighted score
    weighted_score = calculate_dimension_score(component_results)
    
    return weighted_score, component_results

@requires_client
def assess_context_dependency(question_text, passage_text, client=None):
    """
    Assess how dependent the question is on understanding the passage.
    Uses an LLM to determine if the question is self-contained (easy) or highly dependent on the passage (hard).
    
    Args:
        question_text: The text of the question to evaluate
        passage_text: The text of the passage associated with the question (if any)
        client: Anthropic client instance (optional, can be None for unit testing)
        
    Returns:
        int: Binary difficulty score: 0 (easy/self-contained) or 1 (hard/context-dependent)
        dict: Detailed assessment information
        
    Note:
        This function contributes 30% to the context and ambiguity dimension score.
    """
    # Return placeholder values if client is None (for testing)
    if client is None:
        return 0, {"explanation": "Client not provided, returning default score"}
    
    # If no passage is provided, the question is self-contained by default
    if not passage_text:
        return 0, {
            "score": 0,
            "explanation": "No passage provided, question is self-contained by default",
            "context_dependency": "none"
        }
    
    # Construct prompt for the LLM
    prompt = f"""
You are evaluating how dependent a question is on understanding an accompanying passage.
Your task is to determine if the question is SELF-CONTAINED (EASY) or CONTEXT-DEPENDENT (HARD).

Make a binary assessment: 0 for EASY (self-contained), 1 for HARD (context-dependent).

Use these criteria for your assessment:

EASY (Score 0):
- Self-contained, minimal context needed
- Can be answered without referring to the passage
- Contains all necessary information within the question itself
- Passage might provide additional details but isn't essential
- General knowledge is sufficient to answer the question

HARD (Score 1):
- Highly dependent on understanding broader context
- Cannot be meaningfully answered without the passage
- Requires specific information only found in the passage
- Refers to the passage explicitly or implicitly
- Requires integrating information across different parts of the passage

Please analyze the following question and passage:

QUESTION: {question_text}

PASSAGE: {passage_text}

Provide your assessment in JSON format with the following structure:
```json
{{
  "score": 0 or 1,
  "context_dependency": "none/low/medium/high/complete",
  "specific_references": ["List of any explicit references to the passage in the question"],
  "key_information": ["Critical information from the passage needed to answer the question"],
  "reasoning": "Your step-by-step analysis of the context dependency",
  "explanation": "Summary of why you assigned this score"
}}
```
Focus your evaluation specifically on how dependent the question is on the provided context, not other aspects of difficulty.
"""
    try:
        # Get response from the LLM
        response = generate_llm_response(client, prompt)
        # Extract and parse the JSON response
        result = extract_json_from_response(response)
        
        # Validate the result has required fields
        if not all(k in result for k in ["score", "explanation"]):
            raise ValueError("LLM response missing required fields")
        
        # Ensure the score is binary (0 or 1)
        score = 1 if result["score"] > 0.5 else 0
        if result["score"] not in [0, 1]:
            result["original_score"] = result["score"]
            result["score"] = score
            result["explanation"] += " (Score converted to binary value)"
        
        return score, result

    except Exception as e:
        print(f"Error in assess_context_dependency: {str(e)}")
        # Return a default score in case of error
        return 0, {
            "score": 0,
            "explanation": f"Error in evaluation: {str(e)}",
            "error": True
        }

@requires_client
def assess_context_clarity(question_text, passage_text, client=None):
    """
    Assess if the context of the question is clear or if there are ambiguities.
    Uses an LLM to determine if the context is clear and explicit (easy) or ambiguous and implicit (hard).
    
    Args:
        question_text: The text of the question to evaluate
        passage_text: The text of the passage associated with the question (if any)
        client: Anthropic client instance (optional, can be None for unit testing)
        
    Returns:
        int: Binary difficulty score: 0 (easy/clear context) or 1 (hard/ambiguous context)
        dict: Detailed assessment information
        
    Note:
        This function contributes 30% to the context and ambiguity dimension score.
    """
    # Return placeholder values if client is None (for testing)
    if client is None:
        return 0, {"explanation": "Client not provided, returning default score"}
    
    # Combine question and passage text for context evaluation
    context_to_evaluate = question_text
    passage_provided = bool(passage_text)
    if passage_provided:
        context_to_evaluate = f"Question: {question_text}\n\nPassage: {passage_text}"
    
    # Construct prompt for the LLM
    prompt = f"""
You are evaluating the clarity of the context in a question and its associated passage (if provided).
Your task is to determine if the context is CLEAR AND EXPLICIT (EASY) or AMBIGUOUS AND IMPLICIT (HARD).

Make a binary assessment: 0 for EASY (clear context), 1 for HARD (ambiguous context).

Use these criteria for your assessment:

EASY (Score 0):
- Clear, explicit context provided
- Information is presented directly and straightforwardly
- Language is specific and precise
- Terms, names, and references are well-defined
- Relationships between ideas are clearly established
- No significant gaps in information that require inference

HARD (Score 1):
- Ambiguous or implicit context
- Information must be inferred rather than directly stated
- Vague or imprecise language creates multiple possible interpretations
- Undefined terms or unclear references
- Relationships between ideas are implied but not explicitly established
- Requires substantial "reading between the lines"

Please analyze the following:

{context_to_evaluate}

Provide your assessment in JSON format with the following structure:
```json
{{
  "score": 0 or 1,
  "clarity_level": "very clear/clear/somewhat ambiguous/ambiguous/very ambiguous",
  "ambiguous_elements": ["List of specific ambiguous elements if any"],
  "missing_information": ["Any critical missing information that creates ambiguity"],
  "reasoning": "Your step-by-step analysis of the context clarity",
  "explanation": "Summary of why you assigned this score"
}}
```
Focus your evaluation specifically on the clarity or ambiguity of the context, not other aspects of difficulty.
"""
    try:
        # Get response from the LLM
        response = generate_llm_response(client, prompt)

        # Extract and parse the JSON response
        result = extract_json_from_response(response)
        
        # Validate the result has required fields
        if not all(k in result for k in ["score", "explanation"]):
            raise ValueError("LLM response missing required fields")
        
        # Ensure the score is binary (0 or 1)
        score = 1 if result["score"] > 0.5 else 0
        if result["score"] not in [0, 1]:
            result["original_score"] = result["score"]
            result["score"] = score
            result["explanation"] += " (Score converted to binary value)"
        
        # Add a note about passage availability
        result["passage_provided"] = passage_provided
        
        return score, result

    except Exception as e:
        print(f"Error in assess_context_clarity: {str(e)}")
        # Return a default score in case of error
        return 0, {
            "score": 0,
            "explanation": f"Error in evaluation: {str(e)}",
            "error": True,
            "passage_provided": passage_provided
        }

def assess_context_difficulty(question_text, passage_text, client=None):
    """
    Assess the difficulty of the passage/context using both algorithmic metrics
    and LLM-based analysis with parallel processing.
    
    Args:
        question_text: The text of the question to evaluate
        passage_text: The text of the passage associated with the question (if any)
        client: Anthropic client instance (used for Claude API-dependent metrics)
        
    Returns:
        int: Binary difficulty score: 0 (easy) or 1 (hard)
        dict: Detailed assessment information
        
    Note:
        This function contributes 40% to the context and ambiguity dimension score.
    """
    # If no passage provided, the difficulty is minimal
    if not passage_text or passage_text.strip() == "":
        return 0, {
            "score": 0,
            "explanation": "No passage provided, context difficulty is minimal.",
            "passage_provided": False
        }
    
    try:
        # Import the analyzer and parallel processor
        from sat_passage_analyzer import SATPassageAnalyzer
        
        # Initialize the analyzer
        analyzer = SATPassageAnalyzer()
        
        # Create a passage object and load it
        passage = {
            'id': 'passage_1',
            'title': 'Context Passage',
            'text': passage_text
        }
        analyzer.passages = [passage]
        
        # Only initialize the Anthropic client in analyzer if we have a valid client
        # and it's not already initialized to avoid unnecessary reinitialization
        if client is not None and not hasattr(analyzer, 'anthropic_client'):
            analyzer.anthropic_client = client
        
        # Calculate basic metrics (non-API dependent)
        analyzer.calculate_flesch_kincaid()
        analyzer.calculate_avg_sentence_length()
        
        # Try to calculate vocabulary metrics
        try:
            analyzer.calculate_vocabulary_difficulty_ratio()
        except Exception as e:
            print(f"Error calculating vocabulary metrics: {e}")
        
        # If client is available, calculate LLM-dependent metrics in parallel
        if client is not None:
            max_workers = 3  # Limit concurrent API calls to avoid rate limiting
            
            try:
                # Import parallel processing functionality
                from parallelized_analyzer import (
                    calculate_lexile_scores_parallel,
                    calculate_subordinate_clauses_parallel,
                    calculate_syntactic_variety_parallel,
                    calculate_structural_inversions_parallel,
                    calculate_embedded_clauses_parallel,
                    calculate_abstraction_level_parallel,
                    calculate_implied_information_parallel,
                    calculate_argumentative_complexity_parallel,
                    calculate_inference_requirement_parallel, 
                    calculate_figurative_language_parallel
                )
                
                # Dynamically add these methods to our analyzer instance
                import types
                for func_name in [
                    'calculate_lexile_scores_parallel',
                    'calculate_subordinate_clauses_parallel',
                    'calculate_syntactic_variety_parallel',
                    'calculate_structural_inversions_parallel',
                    'calculate_embedded_clauses_parallel',
                    'calculate_abstraction_level_parallel',
                    'calculate_implied_information_parallel',
                    'calculate_argumentative_complexity_parallel',
                    'calculate_inference_requirement_parallel',
                    'calculate_figurative_language_parallel'
                ]:
                    if func_name in globals():
                        setattr(analyzer, func_name, types.MethodType(globals()[func_name], analyzer))
                
                # Run metrics in parallel with max_workers to limit concurrent API calls
                
                
                analyzer.run_with_retry(
                    analyzer.calculate_lexile_scores_parallel,
                    metric_category="readability",
                    metric_name="lexile_score",
                    max_workers=max_workers
                )
                
                analyzer.run_with_retry(
                    analyzer.calculate_subordinate_clauses_parallel,
                    metric_category="syntactic_complexity",
                    metric_name="subordinate_clauses",
                    max_workers=max_workers
                )
                
                analyzer.run_with_retry(
                    analyzer.calculate_syntactic_variety_parallel,
                    metric_category="syntactic_complexity",
                    metric_name="syntactic_variety",
                    max_workers=max_workers
                )

                analyzer.run_with_retry(
                    analyzer.calculate_structural_inversions_parallel,
                    metric_category="syntactic_complexity",
                    metric_name="structural_inversions",
                    max_workers=max_workers
                )
                
                analyzer.run_with_retry(
                    analyzer.calculate_embedded_clauses_parallel,
                    metric_category="syntactic_complexity",
                    metric_name="embedded_clauses",
                    max_workers=max_workers
                )
                
                analyzer.run_with_retry(
                    analyzer.calculate_abstraction_level_parallel,
                    metric_category="conceptual_density",
                    metric_name="abstraction_level",
                    max_workers=max_workers
                )
                
                analyzer.run_with_retry(
                    analyzer.calculate_implied_information_parallel,
                    metric_category="conceptual_density",
                    metric_name="implied_information",
                    max_workers=max_workers
                )
                
                analyzer.run_with_retry(
                    analyzer.calculate_argumentative_complexity_parallel,
                    metric_category="rhetorical_structure",
                    metric_name="argumentative_complexity",
                    max_workers=max_workers
                )

                analyzer.run_with_retry(
                    analyzer.calculate_inference_requirement_parallel,
                    metric_category="cognitive_demands",
                    metric_name="inference_requirement",
                    max_workers=max_workers
                )

                analyzer.run_with_retry(
                    analyzer.calculate_figurative_language_parallel,
                    metric_category="cognitive_demands",
                    metric_name="figurative_language",
                    max_workers=max_workers
                )
            
            except Exception as e:
                print(f"Error initializing or executing parallel processing: {e}")
                # Fall back to sequential processing if parallel processing fails
                try:
                    # Domain-specific terminology complexity
                    analyzer.calculate_lexile_scores()
                    
                    # Syntactic complexity
                    analyzer.calculate_subordinate_clauses()
                    analyzer.calculate_syntactic_variety()
                    analyzer.calculate_structural_inversions()
                    analyzer.calculate_embedded_clauses()
                    
                    # Conceptual density
                    analyzer.calculate_abstraction_level()
                    analyzer.calculate_implied_information()

                    # Rhetorical structure
                    analyzer.calculate_argumentative_complexity()

                    # Cognitive demands
                    analyzer.calculate_inference_requirement()
                    analyzer.calculate_figurative_language()
                except Exception as e:
                    print(f"Error in sequential fallback processing: {e}")
        
        # Calculate overall difficulty
        overall_scores = analyzer.calculate_overall_difficulty()
        
        # Process the results
        if overall_scores and len(overall_scores) > 0:
            # Get the overall score (0-5 scale)
            difficulty_score = overall_scores[0].get("overall_score", 5)
            
            # Determine binary score based on threshold (6.5)
            # 0 for easy (< 6.5), 1 for hard (>= 6.5)
            binary_score = 1 if difficulty_score >= 6.5 else 0
            
            # Compile detailed results
            details = {
                "difficulty_score": difficulty_score,
                "binary_score": binary_score,
                "category_scores": overall_scores[0].get("category_scores", {}),
                "explanation": f"Passage difficulty score is {overall_difficulty:.2f}/10, which is "
                               f"{'above' if binary_score == 1 else 'below'} the threshold of 6.5."
            }
            
            return binary_score, details
        else:
            # Fallback if no results
            return 0, {
                "score": 0,
                "explanation": "Unable to calculate passage difficulty.",
                "error": True
            }
    
    except Exception as e:
        print(f"Error in assess_context_difficulty: {str(e)}")
        # Return a default score in case of error
        return 0, {
            "score": 0,
            "explanation": f"Error in evaluation: {str(e)}",
            "error": True
        }

# Dimension 4: Question Structure and Format (15%)
@requires_client
def aggregate_question_structure(question_data, client=None):
    """
    Function to combine and calculate the metrics for each element in this dimension.
    Takes in the scores of each of the relevant checks in this dimension and combines them based on weight into a dimension score.
    15% of overall score
    """
    question_text = question_data["question"]
    
    # Get scores for each component (placeholders)
    format_score, format_details = evaluate_question_format(question_data)
    visual_score, visual_details = evaluate_visual_aids(question_data)
    clarity_score, clarity_details = evaluate_question_clarity(question_data, client)
    
    # Store all component results
    component_results = {
        "question_format": {
            "score": format_score,
            "weight": STRUCTURE_WEIGHTS["question_format"],
            "details": format_details
        },
        "visual_aids": {
            "score": visual_score,
            "weight": STRUCTURE_WEIGHTS["visual_aids"],
            "details": visual_details
        },
        "question_clarity": {
            "score": clarity_score,
            "weight": STRUCTURE_WEIGHTS["question_clarity"],
            "details": clarity_details
        }
    }
    
    # Calculate weighted score
    weighted_score = calculate_dimension_score(component_results)
    
    return weighted_score, component_results

def evaluate_question_format(question_data):
    """Evaluate what question format is used (mcq, open ended, true/false, etc.). (Algorithm)
    40% of question structure and format score"""
    question_text = question_data["question"]
    passage_text = question_data.get("passage", "")
    options = question_data.get("options", [])
            
    # Convert to lowercase for case-insensitive matching
    text_lower = question_text.lower()
    
    # Check for multiple choice indicators   
    is_multiple_choice = isinstance(options, (list, dict)) and len(options) > 2
    
    # Check for true/false format
    is_true_false = (
    isinstance(options, (list, dict)) and 
    len(options) == 2 and 
    any(
        x == "True" or (isinstance(x, str) and re.match(r"(?i)^yes", x.strip()))
        for x in (options if isinstance(options, list) else options.values())
    )
)

    # Check for matching format
    matching_patterns = [
        r'match.*with', r'match.*to', r'matching',
        r'pair.*with', r'connect.*to', r'link.*with'
    ]
    
    is_matching = any(re.search(pattern, text_lower) for pattern in matching_patterns)
    
    # Check for fill-in-the-blank format
    fill_blank_patterns = [
        r'fill in', r'complete.*(sentence|paragraph|statement)',
        r'___+', r'\.\.\.',  # Underscores or ellipses
        r'\[.*\]', r'<.*>',  # Brackets indicating blanks
        r'\bblank\b'
    ]
    
    is_fill_blank = any(re.search(pattern, text_lower) for pattern in fill_blank_patterns)
    
    # Check for short answer indicators
    short_answer_patterns = [
        r'(brief|short).*(answer|response)', 
        r'answer briefly', 
        r'in (one|two|three|1|2|3|a few) (sentence|word)'
    ]
    
    is_short_answer = any(re.search(pattern, text_lower) for pattern in short_answer_patterns)
    
    # Check for essay or long-form answer indicators
    essay_patterns = [
        r'(write|compose).*(essay|paragraph|composition)',
        r'discuss.*detail', r'elaborate', r'explain.*fully',
        r'analyze and explain', r'in.*words or more'
    ]
    
    is_essay = any(re.search(pattern, text_lower) for pattern in essay_patterns)
    
    # Check for computational or problem-solving indicators
    computational_patterns = [
        r'calculate', r'compute', r'solve', r'find the value',
        r'determine the answer', r'what is the result'
    ]
    
    is_computational = any(re.search(pattern, text_lower) for pattern in computational_patterns)
    
    # Define which formats are considered easy (closed) vs. hard (open)
    # Closed formats (usually considered easier) include multiple choice, true/false, matching, fill-in-blank
    # Open formats (usually considered harder) include short answer, essay, and some computational problems
    
    is_closed_format = is_multiple_choice or is_true_false or is_matching or is_fill_blank
    is_open_format = is_short_answer or is_essay
    
    # Handle computational questions based on complexity indicators
    # Simple calculations are treated as closer to closed format
    # Complex problem-solving is treated as closer to open format
    complex_indicators = [
        r'explain your steps', r'show your work', r'justify',
        r'solve.*problem', r'derive', r'prove', r'demonstrate'
    ]
    
    has_complex_indicators = any(re.search(pattern, text_lower) for pattern in complex_indicators)
    
    if is_computational:
        is_closed_format = is_closed_format or not has_complex_indicators
        is_open_format = is_open_format or has_complex_indicators
    
    # If no specific format is detected, analyze sentence structure to determine if it's likely open-ended
    if not (is_closed_format or is_open_format):
        # Questions starting with these words are typically open-ended
        open_ended_starters = ['explain', 'describe', 'discuss', 'analyze', 'evaluate', 
                               'compare', 'contrast', 'examine', 'interpret', 'justify',
                               'argue', 'assess', 'elaborate', 'illustrate', 'reflect']
        
        # Check if question begins with any open-ended starter words
        is_likely_open_ended = any(text_lower.strip().startswith(starter) for starter in open_ended_starters)
        
        # Check if it has "how" or "why" which often indicate open-ended questions
        has_how_why = bool(re.search(r'^(how|why)\b', text_lower.strip()))
        
        is_open_format = is_likely_open_ended or has_how_why
        is_closed_format = not is_open_format
    
    # Determine final difficulty score (0 for easy/closed, 1 for hard/open)
    difficulty_score = 0 if is_closed_format and not is_open_format else 1
    
    # Determine the specific format for detailed reporting
    format_classification = ''
    if is_multiple_choice:
        format_classification = 'multiple-choice'
    elif is_true_false:
        format_classification = 'true-false'
    elif is_matching:
        format_classification = 'matching'
    elif is_fill_blank:
        format_classification = 'fill-in-blank'
    elif is_computational and not has_complex_indicators:
        format_classification = 'basic-computational'
    elif is_computational and has_complex_indicators:
        format_classification = 'complex-computational'
    elif is_short_answer:
        format_classification = 'short-answer'
    elif is_essay:
        format_classification = 'essay'
    else:
        format_classification = 'open-ended' if is_open_format else 'unclassified'
    
    return difficulty_score, {
        "format_classification": format_classification,
        "is_closed_format": is_closed_format,
        "is_open_format": is_open_format,
        "is_multiple_choice": is_multiple_choice,
        "is_true_false": is_true_false,
        "is_matching": is_matching,
        "is_fill_blank": is_fill_blank,
        "is_computational": is_computational,
        "is_complex_computational": is_computational and has_complex_indicators,
        "is_short_answer": is_short_answer,
        "is_essay": is_essay
    }

def evaluate_visual_aids(question_data):
    """Evaluate whether visual aids are effectively used. (Algorithm)
    30% of question structure and format score"""
    # Convert to lowercase for case-insensitive matching
    question_text = question_data["question"]
    passage_text = question_data.get["passage"]
    text_lower = text_lower = (passage_text + ", " + question_text).lower()
    
    # Check for references to visual elements in the question text
    image_references = bool(re.search(r'\b(figure|image|picture|photo|illustration|<img>)\b', text_lower))
    table_references = bool(re.search(r'\b(table|tabular data|row|column|cell)\b', text_lower))
    chart_references = bool(re.search(r'\b(chart|graph|plot|histogram|bar chart|pie chart|line graph)\b', text_lower))
    diagram_references = bool(re.search(r'\b(diagram|schematic|flowchart|map|blueprint)\b', text_lower))
    
    # Count references to visual elements in question text
    visual_references = sum([image_references, table_references, chart_references, diagram_references])
    
    # Extract specific figure/table references with numbers (e.g., "Figure 1", "Table 2.3")
    specific_references = re.findall(
        r'\b(figure|fig\.|table|chart|graph|diagram)\s*(\d+(?:\.\d+)?)\b', 
        text_lower
    )
    specific_references = [f"{ref_type} {ref_num}" for ref_type, ref_num in specific_references]
    
    # Look for phrases that indicate whether visual analysis is required
    requires_visual_interpretation = bool(re.search(
        r'\b(shown|displayed|depicted|illustrated|presented|'
        r'interpret|analyze|identify|label|'
        r'trend|pattern|point|mark)\b', 
        text_lower
    ))
    
    # Check for spatial or visual reasoning requirements
    spatial_reasoning = bool(re.search(
        r'\b(orientation|position|arrangement|direction|rotation|alignment|perspective|'
        r'distance|proximity|spatial|topographic|layout|configure|rotate|flip|turn)\b',
        text_lower
    ))
    
    # Check for comparison between multiple visual elements
    multi_visual_comparison = bool(re.search(
        r'compare.*(figures|tables|charts|graphs|diagrams|images)|'
        r'difference between.*(figure|table|chart|graph|diagram|image).*(and|with)',
        text_lower
    )) or len(specific_references) > 1
    
    # Now, let's determine if the question is likely to have appropriate visual support
    # This is a guess based on the question text, since we don't have the actual visuals
    
    # Questions with specific figure references are more likely to have appropriate visuals
    has_specific_references = len(specific_references) > 0
    
    # Questions with phrases explicitly directing attention to visuals likely have them
    has_directive_phrases = bool(re.search(
        r'(look at|refer to|examine|'
        r'as seen in)(.*?)(figure|image|table|chart|graph|diagram)',
        text_lower
    ))
    
    # When there are visual references with specific identifiers, assume visuals are present
    likely_has_visuals = has_specific_references or has_directive_phrases
    
    # Define difficulty conditions:
    
    # 1. Hard: Visual elements are referenced without clear indication they're provided
    # (Visual references but no specific numbered references or directive phrases)
    missing_visuals = visual_references > 0 and not likely_has_visuals
    
    # 2. Hard: Asked to perform visual interpretation without clear visual support
    insufficient_visuals = requires_visual_interpretation and not likely_has_visuals
    
    # 3. Hard: Complex visual tasks (comparing multiple visuals, spatial reasoning)
    complex_visual_task = multi_visual_comparison or (spatial_reasoning and visual_references > 0)
    
    # Determine final difficulty score
    is_hard = missing_visuals or insufficient_visuals or complex_visual_task
    difficulty_score = 1 if is_hard else 0
    
    return difficulty_score, {
        "visual_references": visual_references,
        "specific_references": specific_references,
        "likely_has_visuals": likely_has_visuals,
        "requires_visual_interpretation": requires_visual_interpretation,
        "spatial_reasoning": spatial_reasoning,
        "multi_visual_comparison": multi_visual_comparison,
        "missing_visuals": missing_visuals,
        "insufficient_visuals": insufficient_visuals,
        "complex_visual_task": complex_visual_task
    }

@requires_client
def evaluate_question_clarity(question_data, client=None):
    """
    Evaluate whether the question is clear and concise, or ambiguous and open to multiple interpretations.
    Uses an LLM to determine if the question is clearly phrased (easy) or ambiguously phrased (hard).
    
    Args:
        question_text: The text of the question to evaluate
        client: Anthropic client instance (optional, can be None for unit testing)
        
    Returns:
        int: Binary difficulty score: 0 (easy/clear) or 1 (hard/ambiguous)
        dict: Detailed assessment information
        
    Note:
        This function contributes 30% to the question structure and format dimension score.
    """
    # Return placeholder values if client is None (for testing)
    if client is None:
        return 0, {"explanation": "Client not provided, returning default score"}

    question_text = question_data["question"]
    passage_text = question_data.get("passage", "")
    options = question_data.get("options", [])
    
    # Format options string if present
    options_text = ""
    if options:
        if isinstance(options, list):
            options_text = "\nOptions:\n" + "\n".join([f"- {opt}" for opt in options])
        elif isinstance(options, dict):
            options_text = "\nOptions:\n" + "\n".join([f"- {k}: {v}" for k, v in options.items()])
    
    # Construct prompt for the LLM
    prompt = f"""
You are evaluating the clarity and conciseness of a question.
Your task is to determine if the question is CLEAR AND DIRECT (EASY) or AMBIGUOUS AND COMPLEX (HARD).

Make a binary assessment: 0 for EASY (clear question), 1 for HARD (ambiguous question).

Use these criteria for your assessment:

EASY (Score 0):
- Clear, direct phrasing
- Single, obvious interpretation
- Precise language with specific terms
- Logical structure that leads to a clear answer
- Concise without unnecessary information
- Clear task or deliverable expected

HARD (Score 1):
- Complex, potentially ambiguous phrasing
- Multiple possible interpretations
- Vague or imprecise language
- Confusing structure that obscures what is being asked
- Excessively wordy or contains distracting details
- Unclear what response is expected from the question

Please analyze the following question:

QUESTION: {question_text}
{f"PASSAGE: {passage_text}" if passage_text else ""}
{options_text}

Provide your assessment in JSON format with the following structure:
```json
{{
  "score": 0 or 1,
  "clarity_level": "very clear/clear/somewhat ambiguous/ambiguous/very ambiguous",
  "ambiguous_elements": ["Specific parts of the question that create ambiguity or confusion"],
  "possible_interpretations": ["Different ways the question could be interpreted, if ambiguous"],
  "reasoning": "Your step-by-step analysis of the question clarity",
  "explanation": "Summary of why you assigned this score"
}}
```
Focus your evaluation specifically on the clarity of the question's phrasing, not other aspects of difficulty.
"""
    try:
        # Get response from the LLM
        response = generate_llm_response(client, prompt)

        # Extract and parse the JSON response
        result = extract_json_from_response(response)
        
        # Validate the result has required fields
        if not all(k in result for k in ["score", "explanation"]):
            raise ValueError("LLM response missing required fields")
        
        # Ensure the score is binary (0 or 1)
        score = 1 if result["score"] > 0.5 else 0
        if result["score"] not in [0, 1]:
            result["original_score"] = result["score"]
            result["score"] = score
            result["explanation"] += " (Score converted to binary value)"
        
        return score, result

    except Exception as e:
        print(f"Error in evaluate_question_clarity: {str(e)}")
        # Return a default score in case of error
        return 0, {
            "score": 0,
            "explanation": f"Error in evaluation: {str(e)}",
            "error": True
        }

# Dimension 5: Linguistic and Semantic Factors (5%)
@requires_client
def aggregate_linguistic_factors(question_data, client=None):
    """
    Function to combine and calculate the metrics for each element in this dimension.
    Takes in the scores of each of the relevant checks in this dimension and combines them based on weight into a dimension score.
    5% of overall score
    """
    question_text = question_data["question"]
    target_grade = question_data.get("target_grade", 8)  # Default to 8th grade if not specified
    
    # Get scores for each component (placeholders)
    language_complexity_score, language_complexity_details = evaluate_language_complexity(question_text, target_grade)
    term_clarity_score, term_clarity_details = evaluate_term_clarity(question_data, client)
    cultural_references_score, cultural_references_details = evaluate_cultural_references(question_data, client)
    
    # Store all component results
    component_results = {
        "language_complexity": {
            "score": language_complexity_score,
            "weight": LINGUISTIC_WEIGHTS["language_complexity"],
            "details": language_complexity_details
        },
        "term_clarity": {
            "score": term_clarity_score,
            "weight": LINGUISTIC_WEIGHTS["term_clarity"],
            "details": term_clarity_details
        },
        "cultural_references": {
            "score": cultural_references_score,
            "weight": LINGUISTIC_WEIGHTS["cultural_references"],
            "details": cultural_references_details
        }
    }
    
    # Calculate weighted score
    weighted_score = calculate_dimension_score(component_results)
    
    return weighted_score, component_results

def evaluate_language_complexity(question_text, target_grade):
    """Evaluate the language complexity of the question. (Algorithm)
    40% of linguistic and semantic factors score"""
    # Clean and normalize the text
    text = question_text.strip()
    if not text:
        return 0, {"error": "Empty text provided"}
    
    # Count sentences
    # This is a simplified approach that counts sentence-ending punctuation
    sentence_endings = re.findall(r'[.!?]+', text)
    sentence_count = len(sentence_endings)
    if sentence_count == 0:
        sentence_count = 1  # Ensure we don't divide by zero later
    
    # Count words (excluding certain punctuation)
    words = re.findall(r'\b[a-zA-Z0-9\'-]+\b', text)
    word_count = len(words)
    if word_count == 0:
        return 0, {"error": "No words found in text"}
    
    # Count syllables
    def count_syllables(word):
        """Count the number of syllables in a word."""
        word = word.lower()
        if len(word) <= 3:
            return 1
        
        # Remove ending punctuation
        word = re.sub(r'[^a-zA-Z]', '', word)
        
        # Count vowel groups
        vowels = "aeiouy"
        count = 0
        prev_is_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_is_vowel:
                count += 1
            prev_is_vowel = is_vowel
        
        # Adjust for common patterns
        if word.endswith('e'):
            count -= 1
        if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
            count += 1
        if count == 0:
            count = 1
        
        return count
    
    # Calculate total syllables
    syllable_count = sum(count_syllables(word) for word in words)
    
    # Calculate Flesch-Kincaid Grade Level
    # Formula: 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59
    words_per_sentence = word_count / sentence_count
    syllables_per_word = syllable_count / word_count
    flesch_kincaid_grade = 0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59
    
    # Round to one decimal place
    flesch_kincaid_grade = round(flesch_kincaid_grade, 1)
    
    # Compare to target grade
    is_appropriate = flesch_kincaid_grade <= target_grade
    difficulty_score = 0 if is_appropriate else 1  # 0 for easy, 1 for hard
    
    # Calculate grade difference
    grade_difference = flesch_kincaid_grade - target_grade
    
    # Additional metrics for complex sentence structure
    # Count words with more than 2 syllables
    complex_words = [word for word in words if count_syllables(word) > 2]
    complex_word_percentage = len(complex_words) / word_count * 100
    
    # Average word length
    avg_word_length = sum(len(word) for word in words) / word_count
    
    # Long sentences (more than 20 words)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    long_sentences = [s for s in sentences if len(re.findall(r'\b\w+\b', s)) > 20]
    long_sentence_percentage = len(long_sentences) / max(len(sentences), 1) * 100
    
    return difficulty_score, {
        "flesch_kincaid_grade": flesch_kincaid_grade,
        "target_grade": target_grade,
        "grade_difference": grade_difference,
        "is_appropriate": is_appropriate,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "syllable_count": syllable_count,
        "words_per_sentence": words_per_sentence,
        "syllables_per_word": syllables_per_word,
        "complex_word_percentage": complex_word_percentage,
        "avg_word_length": avg_word_length,
        "long_sentence_percentage": long_sentence_percentage
    }

@requires_client
def evaluate_term_clarity(question_data, client=None):
    """
    Evaluate whether the terms and concepts used in the question are clearly defined and understood.
    Uses an LLM to determine if terms are clear and well-defined (easy) or undefined and ambiguous (hard).
    
    Args:
        question_text: The text of the question to evaluate
        client: Anthropic client instance (optional, can be None for unit testing)
        
    Returns:
        int: Binary difficulty score: 0 (easy/clear terms) or 1 (hard/unclear terms)
        dict: Detailed assessment information
        
    Note:
        This function contributes 30% to the linguistic and semantic factors dimension score.
    """
    # Return placeholder values if client is None (for testing)
    if client is None:
        return 0, {"explanation": "Client not provided, returning default score"}

    question_text = question_data["question"]
    passage_text = question_data.get("passage", "")
    options = question_data.get("options", [])
    
    # Format options string if present
    options_text = ""
    if options:
        if isinstance(options, list):
            options_text = "\nOptions:\n" + "\n".join([f"- {opt}" for opt in options])
        elif isinstance(options, dict):
            options_text = "\nOptions:\n" + "\n".join([f"- {k}: {v}" for k, v in options.items()])
    
    # Construct prompt for the LLM
    prompt = f"""
You are evaluating the clarity of terms and concepts used in a question.
Your task is to determine if the terms and concepts are CLEARLY DEFINED (EASY) or UNDEFINED AND AMBIGUOUS (HARD).

Make a binary assessment: 0 for EASY (clear terms), 1 for HARD (unclear terms).

Use these criteria for your assessment:

EASY (Score 0):
- All terms clearly defined or commonly understood
- Concepts are presented in a straightforward manner
- Technical terms are explained or would be familiar to the intended audience
- No ambiguous terms with multiple possible meanings
- No jargon without explanation (unless appropriate for the target audience)
- Key concepts are explicit rather than implicit

HARD (Score 1):
- Undefined or ambiguous terms
- Technical concepts without sufficient explanation
- Terms that could have multiple interpretations in the given context
- Specialized jargon without clarification
- Vague or imprecise language that obscures meaning
- Important concepts that must be inferred rather than being explicitly stated

Please analyze the following question:

QUESTION: {question_text}
{f"PASSAGE: {passage_text}" if passage_text else ""}
{options_text}

Provide your assessment in JSON format with the following structure:
```json
{{
  "score": 0 or 1,
  "key_terms": ["List of important terms and concepts in the question"],
  "undefined_terms": ["Terms that are not clearly defined or could be ambiguous"],
  "technical_jargon": ["Specialized terminology that might not be widely understood"],
  "reasoning": "Your step-by-step analysis of the term clarity",
  "explanation": "Summary of why you assigned this score"
}}
```
Focus your evaluation specifically on the clarity of terms and concepts, not other aspects of difficulty.
"""
    try:
        # Get response from the LLM
        response = generate_llm_response(client, prompt)

        # Extract and parse the JSON response
        result = extract_json_from_response(response)
        
        # Validate the result has required fields
        if not all(k in result for k in ["score", "explanation"]):
            raise ValueError("LLM response missing required fields")
        
        # Ensure the score is binary (0 or 1)
        score = 1 if result["score"] > 0.5 else 0
        if result["score"] not in [0, 1]:
            result["original_score"] = result["score"]
            result["score"] = score
            result["explanation"] += " (Score converted to binary value)"
        
        return score, result

    except Exception as e:
        print(f"Error in evaluate_term_clarity: {str(e)}")
        # Return a default score in case of error
        return 0, {
            "score": 0,
            "explanation": f"Error in evaluation: {str(e)}",
            "error": True
        }

@requires_client
def evaluate_cultural_references(question_data, client=None):
    """
    Evaluate if there are any cultural references or nuances that might affect how learners interpret the question.
    Uses an LLM to determine if the question uses universal references (easy) or specific cultural references (hard).
    
    Args:
        question_text: The text of the question to evaluate
        client: Anthropic client instance (optional, can be None for unit testing)
        
    Returns:
        int: Binary difficulty score: 0 (easy/universal) or 1 (hard/culturally specific)
        dict: Detailed assessment information
        
    Note:
        This function contributes 30% to the linguistic and semantic factors dimension score.
    """
    # Return placeholder values if client is None (for testing)
    if client is None:
        return 0, {"explanation": "Client not provided, returning default score"}

    question_text = question_data["question"]
    passage_text = question_data.get("passage", "")
    options = question_data.get("options", [])
    
    # Format options string if present
    options_text = ""
    if options:
        if isinstance(options, list):
            options_text = "\nOptions:\n" + "\n".join([f"- {opt}" for opt in options])
        elif isinstance(options, dict):
            options_text = "\nOptions:\n" + "\n".join([f"- {k}: {v}" for k, v in options.items()])
    
    # Construct prompt for the LLM
    prompt = f"""
You are evaluating whether a question contains cultural references or nuances that might affect interpretation.
Your task is to determine if the question uses UNIVERSAL REFERENCES (EASY) or SPECIFIC CULTURAL REFERENCES (HARD).

Make a binary assessment: 0 for EASY (universal), 1 for HARD (culturally specific).

Use these criteria for your assessment:

EASY (Score 0):
- Universal references or no cultural references
- Content that would be familiar across diverse backgrounds
- Examples and scenarios that are broadly accessible
- Avoids idioms, slang, or culturally-specific expressions
- Minimal reliance on knowledge of specific cultural contexts
- Would be equally interpretable by people from different cultures

HARD (Score 1):
- Specific cultural references or nuances
- References to traditions, customs, or practices specific to certain cultures
- Use of regional idioms, expressions, or colloquialisms
- Examples that assume familiarity with specific cultural contexts
- Historical or social references that are not universally known
- Content that could be interpreted differently based on cultural background

Please analyze the following question:

QUESTION: {question_text}
{f"PASSAGE: {passage_text}" if passage_text else ""}
{options_text}

Provide your assessment in JSON format with the following structure:
```json
{{
  "score": 0 or 1,
  "cultural_references": ["Specific cultural references identified in the question"],
  "potentially_unfamiliar_elements": ["Elements that might be unfamiliar to some cultural groups"],
  "cultural_knowledge_required": ["Specific cultural knowledge needed to fully understand the question"],
  "reasoning": "Your step-by-step analysis of the cultural references and nuances",
  "explanation": "Summary of why you assigned this score"
}}
```
Focus your evaluation specifically on the presence and impact of cultural references, not other aspects of difficulty.
"""
    try: 
        # Get response from the LLM
        response = generate_llm_response(client, prompt)

        # Extract and parse the JSON response
        result = extract_json_from_response(response)
        
        # Validate the result has required fields
        if not all(k in result for k in ["score", "explanation"]):
            raise ValueError("LLM response missing required fields")
        
        # Ensure the score is binary (0 or 1)
        score = 1 if result["score"] > 0.5 else 0
        if result["score"] not in [0, 1]:
            result["original_score"] = result["score"]
            result["score"] = score
            result["explanation"] += " (Score converted to binary value)"
        
        return score, result

    except Exception as e:
        print(f"Error in evaluate_cultural_references: {str(e)}")
        # Return a default score in case of error
        return 0, {
            "score": 0,
            "explanation": f"Error in evaluation: {str(e)}",
            "error": True
        }

# Dimension 6: Learning Objectives and Assessment (15%)
@requires_client
def aggregate_learning_objectives(question_data, client=None):
    """
    Function to combine and calculate the metrics for each element in this dimension.
    Takes in the scores of each of the relevant checks in this dimension and combines them based on weight into a dimension score.
    5% of overall score
    """
    question_text = question_data["question"]
    
    # Get scores for each component (placeholders)
    bloom_score, bloom_details = determine_bloom_taxonomy_level(question_text, client)
    dok_score, dok_details = determine_dok_level(question_text, client)
    contribution_score, contribution_details = evaluate_contribution_assessment(question_data, client)
    
    # Store all component results
    component_results = {
        "bloom_taxonomy": {
            "score": bloom_score,
            "weight": OBJECTIVES_WEIGHTS["bloom_taxonomy"],
            "details": bloom_details
        },
        "dok_level": {
            "score": dok_score,
            "weight": OBJECTIVES_WEIGHTS["dok_level"],
            "details": dok_details
        },
        "contribution_assessment": {
            "score": contribution_score,
            "weight": OBJECTIVES_WEIGHTS["contribution_assessment"],
            "details": contribution_details
        }
    }
    
    # Calculate weighted score
    weighted_score = calculate_dimension_score(component_results)
    
    return weighted_score, component_results

@requires_client
def determine_bloom_taxonomy_level(question_text, client=None):
    """
    Determine the Bloom's taxonomy level of a question using Claude.
    
    Args:
        question_text: The text of the question to evaluate
        client: Anthropic client instance (optional, can be None for unit testing)
        
    Returns:
        int: Binary difficulty score: 0 (easy/lower-level) or 1 (hard/higher-level)
        dict: Detailed assessment information
        
    Note:
        This function contributes 40% to the learning objectives and assessment dimension score.
        It categorizes questions according to Bloom's taxonomy levels:
        - Lower levels (Remember, Understand): Easy (0)
        - Higher levels (Apply, Analyze, Evaluate, Create): Hard (1)
    """
    # Return placeholder values if client is None (for testing)
    if client is None:
        return 0, {"explanation": "Client not provided, returning default score"}
    
    # Construct prompt for the LLM
    prompt = f"""
You are evaluating a question based on Bloom's Taxonomy of Educational Objectives.
Your task is to determine which level of Bloom's Taxonomy this question primarily targets.

Bloom's Taxonomy has six cognitive levels (from lowest to highest):
1. REMEMBER: Recall facts and basic concepts (e.g., define, list, memorize, name, identify)
2. UNDERSTAND: Explain ideas or concepts (e.g., describe, discuss, explain, locate, recognize)
3. APPLY: Use information in new situations (e.g., apply, demonstrate, solve, use, calculate)
4. ANALYZE: Draw connections among ideas (e.g., analyze, compare, contrast, distinguish, examine)
5. EVALUATE: Justify a position or decision (e.g., appraise, argue, defend, judge, critique)
6. CREATE: Produce new or original work (e.g., design, develop, formulate, create, compose)

After identifying the primary Bloom's level, make a binary assessment:
0 for LOWER-LEVEL (Remember, Understand) - considered EASY
1 for HIGHER-LEVEL (Apply, Analyze, Evaluate, Create) - considered HARD

Please analyze the following question:

QUESTION: {question_text}

Provide your assessment in JSON format with the following structure:
```json
{{
  "primary_bloom_level": "One of: Remember, Understand, Apply, Analyze, Evaluate, Create",
  "bloom_level_number": "Number from 1-6 corresponding to the level",
  "key_verbs": ["List of verbs in the question that indicate this level"],
  "binary_score": 0 or 1,
  "reasoning": "Your step-by-step analysis of the Bloom's taxonomy level",
  "explanation": "Summary of why you assigned this score"
}}
```
Focus your evaluation specifically on the cognitive level required by the question according to Bloom's Taxonomy.
"""
    try:
        # Get response from the LLM
        response = generate_llm_response(client, prompt)

        # Extract and parse the JSON response
        result = extract_json_from_response(response)
        
        # Validate the result has required fields
        if not all(k in result for k in ["binary_score", "explanation"]):
            raise ValueError("LLM response missing required fields")
        
        # Ensure the score is binary (0 or 1)
        score = 1 if result["binary_score"] > 0.5 else 0
        if result["binary_score"] not in [0, 1]:
            result["original_score"] = result["binary_score"]
            result["binary_score"] = score
            result["explanation"] += " (Score converted to binary value)"
        
        return score, result

    except Exception as e:
        print(f"Error in determine_bloom_taxonomy_level: {str(e)}")
        # Return a default score in case of error
        return 0, {
            "score": 0,
            "explanation": f"Error in evaluation: {str(e)}",
            "error": True
        }

@requires_client
def determine_dok_level(question_text, client=None):
    """
    Determine the Depth of Knowledge (DOK) level of a question using Claude.
    
    Args:
        question_text: The text of the question to evaluate
        client: Anthropic client instance (optional, can be None for unit testing)
        
    Returns:
        int: Binary difficulty score: 0 (easy/lower-level) or 1 (hard/higher-level)
        dict: Detailed assessment information
        
    Note:
        This function contributes 40% to the learning objectives and assessment dimension score.
        It categorizes questions according to Webb's Depth of Knowledge levels:
        - Lower levels (DOK 1-2): Easy (0)
        - Higher levels (DOK 3-4): Hard (1)
    """
    # Return placeholder values if client is None (for testing)
    if client is None:
        return 0, {"explanation": "Client not provided, returning default score"}
    
    # Construct prompt for the LLM
    prompt = f"""
You are evaluating a question based on Webb's Depth of Knowledge (DOK) framework.
Your task is to determine which DOK level this question primarily targets.

Webb's Depth of Knowledge has four levels:

1. DOK 1 - RECALL AND REPRODUCTION
   - Recall of facts, terms, concepts, or procedures
   - Following simple procedures or formulas
   - One-step, routine problems
   - Examples: List, define, identify, calculate using a simple formula

2. DOK 2 - SKILLS AND CONCEPTS
   - Use of information or conceptual knowledge
   - Multiple steps requiring decision points
   - Requires deeper knowledge than verbatim recall
   - Examples: Compare/contrast, classify, organize, estimate, make observations

3. DOK 3 - STRATEGIC THINKING
   - Reasoning, planning, using evidence
   - More abstract, complex thinking
   - Justifying with multiple sources or solving non-routine problems
   - Examples: Formulate hypothesis, investigate, cite evidence, develop argument

4. DOK 4 - EXTENDED THINKING
   - Complex reasoning across disciplines
   - Synthesizing information from multiple sources
   - Designing solutions to real-world problems
   - Examples: Design and conduct experiments, analyze multiple solutions, connect concepts across subjects

After identifying the primary DOK level, make a binary assessment:
0 for LOWER-LEVEL (DOK 1-2) - considered EASY
1 for HIGHER-LEVEL (DOK 3-4) - considered HARD

Please analyze the following question:

QUESTION: {question_text}

Provide your assessment in JSON format with the following structure:
```json
{{
  "dok_level": "Number from 1-4",
  "dok_category": "Recall/Skills and Concepts/Strategic Thinking/Extended Thinking",
  "key_indicators": ["List of aspects of the question that indicate this DOK level"],
  "binary_score": 0 or 1,
  "reasoning": "Your step-by-step analysis of the DOK level",
  "explanation": "Summary of why you assigned this score"
}}
```
Focus your evaluation specifically on the cognitive complexity and depth of knowledge required by the question.
"""
    try:
        # Get response from the LLM
        response = generate_llm_response(client, prompt)

        # Extract and parse the JSON response
        result = extract_json_from_response(response)
        
        # Validate the result has required fields
        if not all(k in result for k in ["binary_score", "explanation"]):
            raise ValueError("LLM response missing required fields")
        
        # Ensure the score is binary (0 or 1)
        score = 1 if result["binary_score"] > 0.5 else 0
        if result["binary_score"] not in [0, 1]:
            result["original_score"] = result["binary_score"]
            result["binary_score"] = score
            result["explanation"] += " (Score converted to binary value)"
        
        return score, result

    except Exception as e:
        print(f"Error in determine_dok_level: {str(e)}")
        # Return a default score in case of error
        return 0, {
            "score": 0,
            "explanation": f"Error in evaluation: {str(e)}",
            "error": True
        }

@requires_client
def evaluate_contribution_assessment(question_data, client=None):
    """
    Evaluate how the question contributes to general overall assessment goals.
    Uses an LLM to determine if the question focuses on basic knowledge verification (easy) 
    or deep understanding/application assessment (hard).
    
    Args:
        question_text: The text of the question to evaluate
        client: Anthropic client instance (optional, can be None for unit testing)
        
    Returns:
        int: Binary difficulty score: 0 (easy/basic assessment) or 1 (hard/deep assessment)
        dict: Detailed assessment information
        
    Note:
        This function contributes 20% to the learning objectives and assessment dimension score.
    """
    # Return placeholder values if client is None (for testing)
    if client is None:
        return 0, {"explanation": "Client not provided, returning default score"}

    question_text = question_data["question"]
    passage_text = question_data.get("passage", "")
    options = question_data.get("options", [])
    
    # Format options string if present
    options_text = ""
    if options:
        if isinstance(options, list):
            options_text = "\nOptions:\n" + "\n".join([f"- {opt}" for opt in options])
        elif isinstance(options, dict):
            options_text = "\nOptions:\n" + "\n".join([f"- {k}: {v}" for k, v in options.items()])
    
    # Construct prompt for the LLM
    prompt = f"""
You are evaluating how a question contributes to overall assessment goals in an educational context.
Your task is to determine if the question focuses on BASIC KNOWLEDGE VERIFICATION (EASY) or DEEP UNDERSTANDING/APPLICATION (HARD).

Make a binary assessment: 0 for EASY (basic knowledge verification), 1 for HARD (deep understanding or application).

Use these criteria for your assessment:

EASY (Score 0):
- Basic knowledge verification
- Focuses on recall of facts or information
- Tests recognition or simple comprehension
- Confirms whether fundamental concepts are understood
- Primarily assesses lower-level educational objectives
- Limited connection to real-world application
- Could be answered with rote learning or memorization

HARD (Score 1):
- Deep understanding or application assessment
- Requires demonstration of thorough comprehension
- Tests ability to apply knowledge in complex scenarios
- Assesses critical thinking and analytical skills
- Evaluates higher-level educational objectives
- Connects to real-world applications or authentic contexts
- Requires synthesis of multiple concepts or skills
- Focuses on transferable skills rather than isolated facts

Please analyze the following question:

QUESTION: {question_text}
{f"PASSAGE: {passage_text}" if passage_text else ""}
{options_text}

Provide your assessment in JSON format with the following structure:
```json
{{
  "score": 0 or 1,
  "assessment_purpose": "The primary purpose this question serves in an assessment",
  "skills_assessed": ["Skills or competencies being evaluated by this question"],
  "educational_value": "Evaluation of the question's contribution to learning objectives",
  "reasoning": "Your step-by-step analysis of the question's assessment contribution",
  "explanation": "Summary of why you assigned this score"
}}
```
Focus your evaluation specifically on how the question contributes to assessment goals, not other aspects of difficulty.
"""
    try:
        # Get response from the LLM
        response = generate_llm_response(client, prompt)

        # Extract and parse the JSON response
        result = extract_json_from_response(response)
        
        # Validate the result has required fields
        if not all(k in result for k in ["score", "explanation"]):
            raise ValueError("LLM response missing required fields")
        
        # Ensure the score is binary (0 or 1)
        score = 1 if result["score"] > 0.5 else 0
        if result["score"] not in [0, 1]:
            result["original_score"] = result["score"]
            result["score"] = score
            result["explanation"] += " (Score converted to binary value)"
        
        return score, result

    except Exception as e:
        print(f"Error in evaluate_contribution_assessment: {str(e)}")
        # Return a default score in case of error
        return 0, {
            "score": 0,
            "explanation": f"Error in evaluation: {str(e)}",
            "error": True
        }

# Aggregation Functions
def calculate_dimension_score(component_results):
    """Calculate the weighted score for a single dimension."""
    total_score = 0
    total_weight = 0
    
    for component, result in component_results.items():
        score = result["score"]
        weight = result["weight"]
        total_score += score * weight
        total_weight += weight
    
    return total_score / total_weight if total_weight > 0 else 0

def calculate_overall_difficulty(dimension_scores):
    """Calculate the overall difficulty score from all dimensions."""
    total_score = 0
    total_weight = 0
    
    # Check that all expected dimensions are present
    expected_dimensions = set(DIMENSION_WEIGHTS.keys())
    actual_dimensions = set(dimension_scores.keys())
    missing_dimensions = expected_dimensions - actual_dimensions
    
    if missing_dimensions:
        print(f"Warning: Missing dimensions in calculation: {missing_dimensions}")
    
    for dimension, score in dimension_scores.items():
        weight = DIMENSION_WEIGHTS.get(dimension, 0)
        total_score += score * weight
        total_weight += weight
    
    return total_score / total_weight if total_weight > 0 else 0

def determine_difficulty_level(score):
    """Map a numerical score to a difficulty level."""
    for (lower, upper), level in DIFFICULTY_LEVELS.items():
        if lower <= score <= upper:
            return level
    return "Unknown"

# Output and Visualization Functions
def generate_difficulty_report(question_results, overall_result, output_format="markdown", metadata, output_file=None):
    """
    Generate a comprehensive difficulty report for question assessment results.
    
    Args:
        question_results: List of dictionaries containing difficulty assessment results for each question
        overall_result: Dictionary containing overall test difficulty metrics
        output_format: Format for the report output - "text", "markdown", or "html" (default: "markdown")
        output_file: Path to save the report (if None, returns the report as a string)
        
    Returns:
        String containing the formatted report if output_file is None, otherwise None
    """
    # Import necessary libraries
    from datetime import datetime
    import statistics
    
    # Set up the appropriate formatting functions based on output_format
    if output_format == "html":
        h1 = lambda text: f"<h1>{text}</h1>"
        h2 = lambda text: f"<h2>{text}</h2>"
        h3 = lambda text: f"<h3>{text}</h3>"
        bold = lambda text: f"<strong>{text}</strong>"
        italic = lambda text: f"<em>{text}</em>"
        paragraph = lambda text: f"<p>{text}</p>"
        linebreak = "<br>"
        hr = "<hr>"
        
        def create_table(headers, rows):
            """Create an HTML table with headers and rows."""
            html = "<table border='1' cellpadding='5' cellspacing='0'>\n"
            # Add headers
            html += "  <tr>\n"
            for header in headers:
                html += f"    <th>{header}</th>\n"
            html += "  </tr>\n"
            # Add rows
            for row in rows:
                html += "  <tr>\n"
                for cell in row:
                    html += f"    <td>{cell}</td>\n"
                html += "  </tr>\n"
            html += "</table>"
            return html
        
        doc_start = "<!DOCTYPE html>\n<html>\n<head>\n<title>Question Difficulty Assessment Report</title>\n<style>\n"
        doc_start += "body { font-family: Arial, sans-serif; line-height: 1.6; max-width: 1000px; margin: 0 auto; padding: 20px; }\n"
        doc_start += "h1 { color: #2c3e50; }\n"
        doc_start += "h2 { color: #3498db; margin-top: 30px; }\n"
        doc_start += "h3 { margin-top: 20px; }\n"
        doc_start += "table { border-collapse: collapse; width: 100%; margin: 20px 0; }\n"
        doc_start += "th { background-color: #f2f2f2; }\n"
        doc_start += "th, td { text-align: left; padding: 12px; }\n"
        doc_start += "tr:nth-child(even) { background-color: #f9f9f9; }\n"
        doc_start += ".very-easy { background-color: #d4efdf; }\n"
        doc_start += ".easy { background-color: #a9dfbf; }\n"
        doc_start += ".moderate { background-color: #f9e79f; }\n"
        doc_start += ".hard { background-color: #f5b7b1; }\n"
        doc_start += ".very-hard { background-color: #ec7063; }\n"
        doc_start += ".highlight { background-color: #e8f8f5; padding: 15px; border-left: 5px solid #3498db; margin: 20px 0; }\n"
        doc_start += "</style>\n</head>\n<body>\n"
        doc_end = "</body>\n</html>"
        
        def color_difficulty(level, text):
            """Apply color classes based on difficulty level."""
            css_class = level.lower().replace(" ", "-")
            return f"<span class='{css_class}'>{text}</span>"
        
        def highlight_box(text):
            """Create a highlighted box for important information."""
            return f"<div class='highlight'>{text}</div>"
            
    elif output_format == "markdown":
        h1 = lambda text: f"# {text}\n"
        h2 = lambda text: f"## {text}\n"
        h3 = lambda text: f"### {text}\n"
        bold = lambda text: f"**{text}**"
        italic = lambda text: f"*{text}*"
        paragraph = lambda text: f"{text}\n\n"
        linebreak = "\n"
        hr = "---\n"
        
        def create_table(headers, rows):
            """Create a markdown table with headers and rows."""
            # Header row
            table = "| " + " | ".join(headers) + " |\n"
            # Separator row
            table += "| " + " | ".join(["---" for _ in headers]) + " |\n"
            # Data rows
            for row in rows:
                table += "| " + " | ".join([str(cell) for cell in row]) + " |\n"
            return table + "\n"
        
        doc_start = ""
        doc_end = ""
        
        def color_difficulty(level, text):
            """No colors in markdown, so just add the level as text."""
            return f"{text} ({level})"
        
        def highlight_box(text):
            """Create a highlighted box using blockquote in markdown."""
            return "> " + text.replace("\n", "\n> ") + "\n\n"
    
    else:  # Plain text
        h1 = lambda text: f"{text.upper()}\n{'=' * len(text)}\n"
        h2 = lambda text: f"{text}\n{'-' * len(text)}\n"
        h3 = lambda text: f"{text}\n{'~' * len(text)}\n"
        bold = lambda text: f"{text}"  # No bold in plain text
        italic = lambda text: f"{text}"  # No italic in plain text
        paragraph = lambda text: f"{text}\n\n"
        linebreak = "\n"
        hr = "-" * 80 + "\n"
        
        def create_table(headers, rows):
            """Create a plain text table with headers and rows."""
            # Determine column widths (minimum 5 characters)
            col_widths = []
            for i in range(len(headers)):
                header_width = len(str(headers[i]))
                data_width = max([len(str(row[i])) if i < len(row) else 0 for row in rows], default=0)
                col_widths.append(max(header_width, data_width, 5))
            
            # Create the table
            table = ""
            # Header row
            for i, header in enumerate(headers):
                table += str(header).ljust(col_widths[i]) + " | "
            table = table.rstrip(" | ") + "\n"
            
            # Separator row
            for width in col_widths:
                table += "-" * width + "-+-"
            table = table.rstrip("-+-") + "\n"
            
            # Data rows
            for row in rows:
                for i in range(len(headers)):
                    cell = row[i] if i < len(row) else ""
                    table += str(cell).ljust(col_widths[i]) + " | "
                table = table.rstrip(" | ") + "\n"
            
            return table + "\n"
        
        doc_start = ""
        doc_end = ""
        
        def color_difficulty(level, text):
            """No colors in plain text, so just add the level as text."""
            return f"{text} ({level})"
        
        def highlight_box(text):
            """Create a highlighted box in plain text."""
            border = "-" * 80
            return f"{border}\n{text}\n{border}\n\n"
    
    # Start building the report
    report = doc_start
    
    # Report header
    report += h1("Question Difficulty Assessment Report")
    report += paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report += hr
    
    # ------------------------------------------------------------------------
    # 1. EXECUTIVE SUMMARY
    # ------------------------------------------------------------------------
    report += h2("Executive Summary")
    
    # Overall test difficulty
    score = overall_result["score"]
    level = overall_result["level"]
    report += paragraph(f"The overall test difficulty is {bold(color_difficulty(level, f'{score:.2f}'))} ({level}).")
    
    # Distribution of question difficulties
    difficulty_counts = {}
    for result in question_results:
        level = result["difficulty_level"]
        difficulty_counts[level] = difficulty_counts.get(level, 0) + 1
    
    report += h3("Question Difficulty Distribution")
    
    # Create a table for difficulty distribution
    distribution_rows = []
    total_questions = len(question_results)
    
    # Ensure all difficulty levels are represented in order
    all_levels = ["Very Easy", "Easy", "Moderate", "Hard", "Very Hard"]
    for level in all_levels:
        count = difficulty_counts.get(level, 0)
        percentage = (count / total_questions) * 100 if total_questions > 0 else 0
        distribution_rows.append([level, count, f"{percentage:.1f}%"])
    
    report += create_table(["Difficulty Level", "Count", "Percentage"], distribution_rows)
    
    # Identify standout dimensions
    dimension_averages = overall_result["dimension_averages"]
    sorted_dimensions = sorted(
        dimension_averages.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    highest_dim, highest_score = sorted_dimensions[0]
    lowest_dim, lowest_score = sorted_dimensions[-1]
    
    # Format dimension names for display
    format_dimension_name = lambda name: name.replace("_", " ").title()
    
    report += h3("Key Contributing Dimensions")
    
    report += paragraph(
        f"The dimension contributing most to difficulty is {bold(format_dimension_name(highest_dim))} " +
        f"with an average score of {bold(f'{highest_score:.2f}')}."
    )
    
    report += paragraph(
        f"The dimension contributing least to difficulty is {bold(format_dimension_name(lowest_dim))} " +
        f"with an average score of {bold(f'{lowest_score:.2f}')}."
    )
    
    # Quick recommendations based on overall difficulty
    report += h3("Initial Recommendations")
    
    if score < 0.3:
        recommendation = "The test appears to be quite easy. Consider adding more challenging questions, " + \
                         "particularly those requiring higher-order thinking or deeper knowledge application."
    elif score > 0.7:
        recommendation = "The test appears to be quite difficult. Consider balancing with more accessible questions " + \
                         "or reviewing the most challenging items to ensure they align with learning objectives."
    else:
        recommendation = "The test appears to have a good balance of difficulty. Review individual questions " + \
                         "that may be outliers to ensure they align with your assessment goals."
    
    report += highlight_box(recommendation)
    
    # ------------------------------------------------------------------------
    # 2. DIMENSION ANALYSIS
    # ------------------------------------------------------------------------
    report += h2("Dimension Analysis")
    
    # Create a table showing all dimensions and their scores
    dimension_rows = []
    for dim_name, dim_score in sorted_dimensions:
        formatted_name = format_dimension_name(dim_name)
        
        # Determine the contribution level based on the score
        contrib_level = "High" if dim_score > 0.6 else "Medium" if dim_score > 0.4 else "Low"
        
        # Use global constants for dimension weights
        weight = DIMENSION_WEIGHTS.get(dim_name, 0)
        
        weighted_contrib = dim_score * weight
        weighted_pct = (weighted_contrib / score) * 100 if score > 0 else 0
        
        dimension_rows.append([
            formatted_name,
            f"{dim_score:.2f}",
            contrib_level,
            f"{weight:.2f}",
            f"{weighted_pct:.1f}%"
        ])
    
    report += create_table(
        ["Dimension", "Average Score", "Contribution Level", "Weight", "% of Total Difficulty"],
        dimension_rows
    )
    
    # Detailed analysis of dimensions
    for dim_name, dim_score in sorted_dimensions:
        formatted_name = format_dimension_name(dim_name)
        report += h3(f"{formatted_name} ({dim_score:.2f})")
        
        # Provide specific insights based on dimension
        if dim_name == "knowledge_requirements":
            if dim_score > 0.7:
                report += paragraph(
                    "Questions require significant prior and domain-specific knowledge. " +
                    "Consider whether this aligns with the expected preparation level of test-takers."
                )
            elif dim_score < 0.3:
                report += paragraph(
                    "Questions require minimal specialized knowledge. " +
                    "Consider if the assessment adequately tests mastery of subject material."
                )
            else:
                report += paragraph(
                    "Questions show a balanced requirement for subject knowledge."
                )
        
        elif dim_name == "cognitive_complexity":
            if dim_score > 0.7:
                report += paragraph(
                    "Questions demand high-level thinking skills including analysis, evaluation, and creation. " +
                    "Ensure test-takers have been prepared for this level of cognitive challenge."
                )
            elif dim_score < 0.3:
                report += paragraph(
                    "Questions primarily test recall and basic comprehension. " +
                    "Consider incorporating higher-order thinking skills if appropriate to learning objectives."
                )
            else:
                report += paragraph(
                    "Questions show a good mix of cognitive complexity levels."
                )
        
        elif dim_name == "context_and_ambiguity":
            if dim_score > 0.7:
                report += paragraph(
                    "Questions are highly context-dependent and may contain ambiguities. " +
                    "Review to ensure clarity and that all necessary context is provided."
                )
            elif dim_score < 0.3:
                report += paragraph(
                    "Questions are straightforward with little contextual dependency. " +
                    "Consider if more complex, context-rich scenarios would better test application."
                )
            else:
                report += paragraph(
                    "Questions show appropriate use of context and minimal problematic ambiguity."
                )
        
        elif dim_name == "question_structure":
            if dim_score > 0.7:
                report += paragraph(
                    "Question structure and format increase difficulty. " +
                    "Consider if the format itself is creating unintended barriers."
                )
            elif dim_score < 0.3:
                report += paragraph(
                    "Question structure and format are straightforward. " +
                    "Consider if more varied question formats would better assess learning."
                )
            else:
                report += paragraph(
                    "Questions use appropriate structures and formats."
                )
        
        elif dim_name == "linguistic_factors":
            if dim_score > 0.7:
                report += paragraph(
                    "Questions use complex language, potentially creating unintended difficulty. " +
                    "Review wording to ensure language isn't a barrier to demonstrating knowledge."
                )
            elif dim_score < 0.3:
                report += paragraph(
                    "Questions use simple, accessible language. " +
                    "This is generally positive but ensure vocabulary aligns with subject mastery expectations."
                )
            else:
                report += paragraph(
                    "Questions show appropriate language complexity for the subject matter."
                )
        
        elif dim_name == "learning_objectives":
            if dim_score > 0.7:
                report += paragraph(
                    "Questions target high-level learning objectives and depth of knowledge. " +
                    "Ensure these align with curriculum goals and instruction."
                )
            elif dim_score < 0.3:
                report += paragraph(
                    "Questions focus on basic learning objectives. " +
                    "Consider if higher DOK levels would better assess mastery."
                )
            else:
                report += paragraph(
                    "Questions align well with a range of learning objectives."
                )
    
    # ------------------------------------------------------------------------
    # 3. QUESTION-BY-QUESTION ANALYSIS
    # ------------------------------------------------------------------------
    report += h2("Question-by-Question Analysis")
    
    # Sort questions by difficulty for better analysis
    sorted_questions = sorted(
        question_results, 
        key=lambda x: x["overall_score"],
        reverse=True
    )
    
    # Table summarizing all questions
    question_rows = []
    for i, result in enumerate(sorted_questions):
        # Truncate long questions for the table
        q_text = result["question_text"]
        truncated_text = (q_text[:60] + "...") if len(q_text) > 60 else q_text
        
        question_rows.append([
            i+1,
            truncated_text,
            f"{result['overall_score']:.2f}",
            result["difficulty_level"]
        ])
    
    report += create_table(
        ["#", "Question (truncated)", "Score", "Difficulty Level"],
        question_rows
    )
    
    # Detailed analysis of each question
    for i, result in enumerate(sorted_questions):
        q_text = result["question_text"]
        score = result["overall_score"]
        level = result["difficulty_level"]
        
        report += h3(f"Question {i+1}: {color_difficulty(level, f'Score: {score:.2f}')}")
        report += paragraph(f"{bold('Question Text:')} {q_text}")
        
        # Dimension breakdown for this question
        dim_rows = []
        for dim_name, dim_score in result["dimension_scores"].items():
            formatted_name = format_dimension_name(dim_name)
            # Determine if this dimension is a key contributor to difficulty
            is_high = dim_score > 0.6
            dim_rows.append([
                formatted_name,
                f"{dim_score:.2f}",
                "High" if is_high else "Low"
            ])
        
        # Sort dimensions by score for this question
        dim_rows.sort(key=lambda x: float(x[1]), reverse=True)
        
        report += paragraph(bold("Dimension Breakdown:"))
        report += create_table(
            ["Dimension", "Score", "Contribution"],
            dim_rows
        )
        
        # Specific improvement suggestions based on the question's profile
        high_dims = [row[0] for row in dim_rows if row[2] == "High"]
        
        if high_dims:
            report += paragraph(bold("Improvement Suggestions:"))
            suggestions = []
            
            for dim in high_dims:
                if "Knowledge Requirements" in dim and score > 0.6:
                    suggestions.append(
                        "Consider whether the specialized knowledge required is aligned with learning objectives and instruction."
                    )
                elif "Cognitive Complexity" in dim and score > 0.6:
                    suggestions.append(
                        "Review if the level of analysis or critical thinking expected is appropriate and scaffolded."
                    )
                elif "Context" in dim and score > 0.6:
                    suggestions.append(
                        "Check if all necessary context is provided and ambiguities are minimized."
                    )
                elif "Linguistic Factors" in dim and score > 0.6:
                    suggestions.append(
                        "Simplify language where possible without compromising content."
                    )
                elif "Question Structure" in dim and score > 0.6:
                    suggestions.append(
                        "Consider if a different question format would assess the same knowledge more effectively."
                    )
            
            if suggestions:
                for sugg in suggestions:
                    report += f"- {sugg}\n"
                report += linebreak
    
    # ------------------------------------------------------------------------
    # 4. PATTERN RECOGNITION
    # ------------------------------------------------------------------------
    report += h2("Pattern Recognition")
    
    # Identify common difficulty factors
    dimension_scores = {dim: [] for dim in DIMENSION_WEIGHTS.keys()}
    for result in question_results:
        for dim, score in result["dimension_scores"].items():
            dimension_scores[dim].append(score)
    
    # Calculate correlation between dimensions
    correlations = {}
    for dim1 in dimension_scores:
        for dim2 in dimension_scores:
            if dim1 >= dim2:  # Only calculate each pair once
                continue
                
            # Calculate correlation coefficient (simplified)
            scores1 = dimension_scores[dim1]
            scores2 = dimension_scores[dim2]
            
            if len(scores1) < 2 or len(scores2) < 2:
                continue
                
            mean1 = statistics.mean(scores1)
            mean2 = statistics.mean(scores2)
            
            numerator = sum((x - mean1) * (y - mean2) for x, y in zip(scores1, scores2))
            denom1 = sum((x - mean1) ** 2 for x in scores1)
            denom2 = sum((y - mean2) ** 2 for y in scores2)
            
            if denom1 == 0 or denom2 == 0:
                correlation = 0
            else:
                correlation = numerator / (denom1 ** 0.5 * denom2 ** 0.5)
                
            # Only keep strong correlations
            if abs(correlation) > 0.5:
                correlations[(dim1, dim2)] = correlation
    
    # Report on dimension correlations
    if correlations:
        report += h3("Dimension Correlations")
        report += paragraph("The following dimensions tend to co-occur in difficulty:")
        
        corr_rows = []
        for (dim1, dim2), corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
            name1 = format_dimension_name(dim1)
            name2 = format_dimension_name(dim2)
            corr_type = "Strong Positive" if corr > 0.7 else \
                       "Moderate Positive" if corr > 0.5 else \
                       "Moderate Negative" if corr < -0.5 else \
                       "Strong Negative"
            
            corr_rows.append([name1, name2, f"{corr:.2f}", corr_type])
        
        report += create_table(
            ["Dimension 1", "Dimension 2", "Correlation", "Relationship"],
            corr_rows
        )
        
        # Interpret strongest correlation
        if corr_rows:
            strongest = corr_rows[0]
            if float(strongest[2]) > 0:
                interpretation = (
                    f"Questions difficult in {strongest[0]} tend to also be difficult in {strongest[1]}. " +
                    f"This suggests these aspects may be related in the subject matter or question design."
                )
            else:
                interpretation = (
                    f"Questions difficult in {strongest[0]} tend to be easier in {strongest[1]}. " +
                    f"This shows an interesting trade-off in the assessment design."
                )
            
            report += paragraph(interpretation)
    else:
        report += paragraph("No strong correlations were found between difficulty dimensions.")
    
    # Identify outlier questions
    report += h3("Outlier Questions")
    
    # Find questions that have unusual difficulty profiles
    outliers = []
    for i, result in enumerate(question_results):
        # Check if any dimension is very different from the overall average
        unusual_dims = []
        for dim, score in result["dimension_scores"].items():
            avg_score = dimension_averages[dim]
            if abs(score - avg_score) > 0.3:  # Significant deviation
                unusual_dims.append((dim, score, avg_score))
        
        if unusual_dims:
            outliers.append((i+1, result["question_text"], unusual_dims))
    
    if outliers:
        report += paragraph("The following questions show unusual difficulty patterns:")
        
        for q_num, q_text, unusual_dims in outliers:
            truncated_text = (q_text[:60] + "...") if len(q_text) > 60 else q_text
            report += paragraph(f"{bold(f'Question {q_num}:')} {truncated_text}")
            
            for dim, score, avg in unusual_dims:
                formatted_dim = format_dimension_name(dim)
                direction = "higher" if score > avg else "lower"
                report += f"- {formatted_dim} difficulty is significantly {direction} than average " + \
                          f"({score:.2f} vs. {avg:.2f} average)\n"
            
            report += linebreak
    else:
        report += paragraph("No significant outlier questions were identified.")
    
    # ------------------------------------------------------------------------
    # 5. RECOMMENDATIONS AND CONCLUSIONS
    # ------------------------------------------------------------------------
    report += h2("Recommendations and Conclusions")
    
    # Overall recommendations based on the full analysis
    report += h3("Assessment Balance")
    
    # Analyze the distribution of question difficulties
    easy_count = sum(1 for r in question_results if r["difficulty_level"] in ["Very Easy", "Easy"])
    mod_count = sum(1 for r in question_results if r["difficulty_level"] == "Moderate")
    hard_count = sum(1 for r in question_results if r["difficulty_level"] in ["Hard", "Very Hard"])
    
    easy_pct = (easy_count / total_questions) * 100 if total_questions > 0 else 0
    mod_pct = (mod_count / total_questions) * 100 if total_questions > 0 else 0
    hard_pct = (hard_count / total_questions) * 100 if total_questions > 0 else 0
    
    # Build recommendations based on distribution
    balance_recommendations = []
    
    if easy_pct < 20:
        balance_recommendations.append(
            f"The assessment has few easy questions ({easy_pct:.1f}%). Consider adding more accessible " +
            "questions to build confidence and establish baseline knowledge."
        )
    
    if hard_pct < 20:
        balance_recommendations.append(
            f"The assessment has few challenging questions ({hard_pct:.1f}%). Consider adding more " +
            "difficult questions to differentiate high-performing students."
        )
    
    if mod_pct < 30:
        balance_recommendations.append(
            f"The assessment has few moderate questions ({mod_pct:.1f}%). A strong middle tier of " +
            "questions helps provide reliable assessment for the majority of students."
        )
    
    if abs(easy_pct - hard_pct) > 30:
        balance_recommendations.append(
            f"There is a significant imbalance between easy ({easy_pct:.1f}%) and hard ({hard_pct:.1f}%) " +
            "questions. A more balanced distribution may provide better assessment across skill levels."
        )
    
    if not balance_recommendations:
        balance_recommendations.append(
            "The assessment shows good balance across difficulty levels. Maintain this balance " +
            "while addressing specific question improvements noted earlier."
        )
    
    for rec in balance_recommendations:
        report += paragraph(rec)
    
    # Dimension-specific recommendations
    report += h3("Dimension-Specific Improvements")
    
    # Identify the top two most challenging dimensions
    top_challenging_dims = sorted_dimensions[:2]
    
    for dim_name, dim_score in top_challenging_dims:
        formatted_name = format_dimension_name(dim_name)
        report += paragraph(f"{bold(formatted_name)} ({dim_score:.2f}):")
        
        if dim_name == "knowledge_requirements":
            report += "- Review whether prerequisite knowledge is clearly communicated to test-takers\n"
            report += "- Consider providing reference materials for specialized knowledge if appropriate\n"
            report += "- Ensure knowledge requirements align with curriculum and instruction\n"
        
        elif dim_name == "cognitive_complexity":
            report += "- Check that higher-order thinking skills have been taught and practiced\n"
            report += "- Consider providing scaffolding questions that build to complex problems\n"
            report += "- Ensure cognitive demands align with learning objectives\n"
        
        elif dim_name == "context_and_ambiguity":
            report += "- Review questions for clarity and sufficient context\n"
            report += "- Consider whether context complexity serves assessment goals\n"
            report += "- Eliminate unintentional ambiguities that don't serve assessment purposes\n"
        
        elif dim_name == "question_structure":
            report += "- Review question formats for clarity and accessibility\n"
            report += "- Consider if visual supports would improve understanding\n"
            report += "- Ensure question structure doesn't introduce unintended difficulty\n"
        
        elif dim_name == "linguistic_factors":
            report += "- Review language complexity to ensure it matches target audience\n"
            report += "- Consider simplifying sentence structures without reducing content rigor\n"
            report += "- Ensure cultural references are accessible to all test-takers\n"
        
        elif dim_name == "learning_objectives":
            report += "- Verify alignment between questions and stated learning objectives\n"
            report += "- Consider if the depth of knowledge required is appropriate\n"
            report += "- Ensure balanced coverage of all relevant learning objectives\n"
        
        report += linebreak
    
    # Final conclusion
    report += h3("Conclusion")
    
    report += highlight_box(
        f"This assessment has an overall difficulty score of {bold(f'{score:.2f}')} ({level}). " +
        f"The primary contributors to difficulty are {bold(format_dimension_name(highest_dim))} " +
        f"and {bold(format_dimension_name(sorted_dimensions[1][0]))}. " +
        "By addressing the recommendations above, you can refine the assessment to better align " +
        "with your educational objectives while maintaining appropriate challenge for your students."
    )
    
    # Add glossary if in formal formats
    if output_format in ["markdown", "html"]:
        report += h2("Glossary of Terms")
        
        glossary_rows = [
            ["Dimension", "An aspect of question difficulty that is evaluated"],
            ["Knowledge Requirements", "Prior and domain knowledge needed to answer the question"],
            ["Cognitive Complexity", "Level of thinking and mental processing required"],
            ["Context and Ambiguity", "Contextual understanding needs and clarity of the question"],
            ["Linguistic Factors", "Language complexity and semantic elements that affect difficulty"],
            ["Question Structure", "Format and presentation elements that impact accessibility"],
            ["Learning Objectives", "Alignment with educational goals and assessment targets"],
            ["DOK", "Depth of Knowledge - framework for cognitive complexity (Levels 1-4)"],
            ["Bloom's Taxonomy", "Hierarchical classification of cognitive learning objectives"]
        ]
        
        report += create_table(["Term", "Definition"], glossary_rows)
    
    # Finish the report
    report += doc_end
    
    # Either save to file or return as string
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"Report successfully saved to {output_file}")
            return None
        except Exception as e:
            print(f"Error saving report to {output_file}: {str(e)}")
            # Fall through to return the report as string
    
    return report

def save_results_to_json(results, output_file):
    """
    Save the difficulty assessment results to a JSON file.
    
    Args:
        results: Dictionary containing the assessment results
        output_file: Path to save the JSON file
        
    Returns:
        None
    """
    import json
    import os
    
    try:
        # Ensure the directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Write with proper formatting and encoding
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Results successfully saved to {output_file}")
        
    except Exception as e:
        print(f"Error saving results to {output_file}: {str(e)}")
        raise

def create_difficulty_visualizations(results, output_dir):
    """
    Create visualizations of the difficulty assessment results.
    
    Args:
        results: Dictionary containing the assessment results (from assess_question_difficulty)
        output_dir: Directory to save the visualization files
        
    Returns:
        None
    """

    # Check for required visualization libraries
    missing_libraries = []
    try:
        import numpy as np
    except ImportError:
        missing_libraries.append("numpy")
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        missing_libraries.append("matplotlib")
    
    try:
        import seaborn as sns
    except ImportError:
        missing_libraries.append("seaborn")
    
    # If any libraries are missing, inform the user and exit
    if missing_libraries:
        print(f"Warning: Cannot create visualizations. Missing libraries: {', '.join(missing_libraries)}")
        print("Please install these libraries using: pip install " + " ".join(missing_libraries))
        return None
    
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Extract data
    questions = results["questions"]
    overall_test_difficulty = results["overall_test_difficulty"]
    
    # -----------------------------------------------------------------
    # 1. OVERALL DIFFICULTY DISTRIBUTION
    # -----------------------------------------------------------------
    
    # Get difficulty scores
    difficulty_scores = [q["overall_score"] for q in questions]
    
    # Create histogram of difficulty scores
    plt.figure(figsize=(10, 6))
    
    # Define custom colormap from green (easy) to red (hard)
    colors = [(0.2, 0.8, 0.2), (0.8, 0.8, 0.2), (0.8, 0.4, 0.2)]  # green to yellow to red
    cmap_name = 'difficulty_colormap'
    cm_difficulty = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)
    
    # Plot histogram with color gradient
    n, bins, patches = plt.hist(difficulty_scores, bins=20, alpha=0.7, edgecolor='black')
    
    # Set colors based on bin values
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    col = bin_centers - min(bin_centers)
    col /= max(col)
    
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm_difficulty(c))
    
    # Add vertical line for overall average
    avg_difficulty = overall_test_difficulty["score"]
    plt.axvline(x=avg_difficulty, color='navy', linestyle='--', linewidth=2, 
                label=f'Overall Average ({avg_difficulty:.2f})')
    
    # Add difficulty level regions
    difficulty_regions = [
        (0.00, 0.20, 'Very Easy', '#d4efdf'),
        (0.21, 0.40, 'Easy', '#a9dfbf'),
        (0.41, 0.60, 'Moderate', '#f9e79f'),
        (0.61, 0.80, 'Hard', '#f5b7b1'),
        (0.81, 1.00, 'Very Hard', '#ec7063')
    ]
    
    for lower, upper, label, color in difficulty_regions:
        plt.axvspan(lower, upper, alpha=0.2, color=color)
        # Add text label in the middle of each region
        plt.text((lower + upper) / 2, max(n) * 0.9, label, 
                 horizontalalignment='center', color='#333333')
    
    plt.title('Distribution of Question Difficulty Scores', fontsize=16)
    plt.xlabel('Difficulty Score', fontsize=12)
    plt.ylabel('Number of Questions', fontsize=12)
    plt.xlim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, 'difficulty_distribution.png'), dpi=300)
    plt.close()
    
    # -----------------------------------------------------------------
    # 2. DIMENSION HEATMAP
    # -----------------------------------------------------------------
    
    # Extract dimension scores for each question
    dimension_data = {}
    all_dimensions = [
        "knowledge_requirements", "cognitive_complexity", "context_and_ambiguity",
        "question_structure", "linguistic_factors", "learning_objectives"
    ]
    
    for i, question in enumerate(questions):
        for dim in all_dimensions:
            if dim not in dimension_data:
                dimension_data[dim] = []
            # Get score or default to 0 if dimension is missing
            score = question["dimension_scores"].get(dim, 0)
            dimension_data[dim].append((i+1, score))
    
    # Format dimension names for display
    format_dimension_name = lambda name: name.replace("_", " ").title()
    dimension_names = [format_dimension_name(dim) for dim in all_dimensions]
    
    # Create the heatmap data
    question_numbers = list(range(1, len(questions) + 1))
    heatmap_data = np.zeros((len(all_dimensions), len(questions)))
    
    for i, dim in enumerate(all_dimensions):
        for j in range(len(questions)):
            # Find the score for question j+1
            for q_num, score in dimension_data[dim]:
                if q_num == j+1:
                    heatmap_data[i, j] = score
                    break
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(
        heatmap_data,
        cmap="YlOrRd",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Difficulty Score"}
    )
    
    # Set labels
    plt.title("Dimension Scores by Question", fontsize=16)
    plt.xlabel("Question Number", fontsize=12)
    plt.ylabel("Dimension", fontsize=12)
    
    # Set tick labels
    ax.set_xticks(np.arange(len(questions)) + 0.5)
    ax.set_xticklabels(question_numbers)
    ax.set_yticks(np.arange(len(all_dimensions)) + 0.5)
    ax.set_yticklabels(dimension_names)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, 'dimension_heatmap.png'), dpi=300)
    plt.close()
    
    # -----------------------------------------------------------------
    # 3. DIMENSION CONTRIBUTION TO OVERALL DIFFICULTY
    # -----------------------------------------------------------------
    
    # Extract dimension averages and weights
    dimension_averages = overall_test_difficulty["dimension_averages"]
    
    # Extract dimension averages and weights
    dimension_averages = overall_test_difficulty["dimension_averages"]
    
    # Use the global constant instead of redefining weights
    dimension_weights = DIMENSION_WEIGHTS
    
    # Calculate weighted contribution of each dimension
    weighted_contributions = {}
    for dim, avg_score in dimension_averages.items():
        weight = dimension_weights.get(dim, 0)
        weighted_contributions[dim] = avg_score * weight
    
    # Sort dimensions by weighted contribution
    sorted_dims = sorted(weighted_contributions.items(), key=lambda x: x[1], reverse=True)
    
    # Prepare data for plotting
    dims = [format_dimension_name(dim) for dim, _ in sorted_dims]
    raw_scores = [dimension_averages[dim] for dim, _ in sorted_dims]
    weighted_scores = [weighted_contributions[dim] for dim, _ in sorted_dims]
    weights = [dimension_weights[dim] for dim, _ in sorted_dims]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Dimension Average Scores
    bars1 = ax1.bar(
        dims, 
        raw_scores, 
        color=plt.cm.YlOrRd(np.array(raw_scores)),
        edgecolor='black',
        alpha=0.7
    )
    
    ax1.set_title('Dimension Average Scores', fontsize=16)
    ax1.set_xlabel('Dimension', fontsize=12)
    ax1.set_ylabel('Average Score', fontsize=12)
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.02,
            f'{height:.2f}',
            ha='center', 
            va='bottom',
            fontsize=9
        )
    
    # Rotate x labels for better readability
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # Plot 2: Weighted Contribution to Overall Difficulty
    bars2 = ax2.bar(
        dims, 
        weighted_scores, 
        color=plt.cm.YlOrRd(np.array(weighted_scores) / max(weighted_scores)),
        edgecolor='black',
        alpha=0.7
    )
    
    ax2.set_title('Weighted Contribution to Overall Difficulty', fontsize=16)
    ax2.set_xlabel('Dimension', fontsize=12)
    ax2.set_ylabel('Weighted Contribution', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        weight_percent = weights[i] * 100
        ax2.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.01,
            f'{height:.2f}\n({weight_percent:.0f}%)',
            ha='center', 
            va='bottom',
            fontsize=9
        )
    
    # Rotate x labels for better readability
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # Add overall difficulty score reference
    ax2.axhline(
        y=overall_test_difficulty["score"], 
        color='navy', 
        linestyle='--', 
        linewidth=2,
        label=f'Overall Score ({overall_test_difficulty["score"]:.2f})'
    )
    ax2.legend()
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, 'dimension_contributions.png'), dpi=300)
    plt.close()
    
    # -----------------------------------------------------------------
    # 4. RADAR CHART OF DIMENSIONS
    # -----------------------------------------------------------------
    
    # Number of dimensions
    N = len(all_dimensions)
    
    # Formatted names for radar chart
    radar_names = [format_dimension_name(dim) for dim in all_dimensions]
    
    # Get dimension averages in the right order
    radar_values = [dimension_averages[dim] for dim in all_dimensions]
    
    # Create angles for each dimension
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    
    # Close the polygon
    radar_values.append(radar_values[0])
    radar_names.append(radar_names[0])
    angles.append(angles[0])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Plot radar chart
    ax.plot(angles, radar_values, 'o-', linewidth=2, color='#e74c3c')
    ax.fill(angles, radar_values, alpha=0.25, color='#e74c3c')
    
    # Set labels
    ax.set_thetagrids(np.degrees(angles[:-1]), radar_names[:-1], fontsize=12)
    
    # Set y limits
    ax.set_ylim(0, 1)
    
    # Add grid lines at 0.2, 0.4, 0.6, 0.8
    grid_values = [0.2, 0.4, 0.6, 0.8]
    grid_labels = ['0.2', '0.4', '0.6', '0.8']
    
    ax.set_rgrids(
        grid_values, 
        labels=grid_labels,
        angle=0, 
        fontsize=10
    )
    
    # Add difficulty level regions
    # Create custom circular patches for the legend
    for lower, upper, label, color in difficulty_regions:
        middle = (lower + upper) / 2
        # Add colored regions
        ax.fill_between(
            np.linspace(0, 2*np.pi, 100),
            lower, upper,
            color=color,
            alpha=0.2
        )
    
    # Add a legend for difficulty levels
    import matplotlib.patches as mpatches
    patches = []
    for _, _, label, color in difficulty_regions:
        patch = mpatches.Patch(color=color, alpha=0.3, label=label)
        patches.append(patch)
    
    ax.legend(
        handles=patches, 
        loc='upper right',
        bbox_to_anchor=(0.1, 0.1)
    )
    
    plt.title('Dimension Profile', fontsize=16, y=1.08)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, 'dimension_radar.png'), dpi=300)
    plt.close()
    
    # -----------------------------------------------------------------
    # 5. QUESTION DIFFICULTY TIMELINE
    # -----------------------------------------------------------------
    
    # Extract difficulty scores in question order
    question_numbers = list(range(1, len(questions) + 1))
    scores = [q["overall_score"] for q in questions]
    levels = [q["difficulty_level"] for q in questions]
    
    # Define colors for each difficulty level
    level_colors = {
        "Very Easy": "#d4efdf",
        "Easy": "#a9dfbf",
        "Moderate": "#f9e79f",
        "Hard": "#f5b7b1",
        "Very Hard": "#ec7063"
    }
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot difficulty scores
    plt.plot(question_numbers, scores, 'o-', linewidth=2, color='#3498db', markersize=8)
    
    # Color each point based on difficulty level
    for i, (q_num, score, level) in enumerate(zip(question_numbers, scores, levels)):
        plt.plot(q_num, score, 'o', markersize=10, color=level_colors.get(level, "#333333"))
    
    # Add average line
    plt.axhline(
        y=overall_test_difficulty["score"],
        color='navy',
        linestyle='--',
        linewidth=2,
        label=f'Average ({overall_test_difficulty["score"]:.2f})'
    )
    
    # Add difficulty level regions
    for lower, upper, label, color in difficulty_regions:
        plt.axhspan(lower, upper, alpha=0.2, color=color)
        # Add text label on the right side
        plt.text(
            len(questions) + 0.5, 
            (lower + upper) / 2, 
            label, 
            verticalalignment='center',
            fontsize=9
        )
    
    plt.title('Question Difficulty Timeline', fontsize=16)
    plt.xlabel('Question Number', fontsize=12)
    plt.ylabel('Difficulty Score', fontsize=12)
    plt.ylim(0, 1)
    plt.xlim(0.5, len(questions) + 0.5)
    plt.xticks(question_numbers)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, 'question_timeline.png'), dpi=300)
    plt.close()
    
    # Return the paths to the created visualizations
    visualization_files = [
        os.path.join(output_dir, 'difficulty_distribution.png'),
        os.path.join(output_dir, 'dimension_heatmap.png'),
        os.path.join(output_dir, 'dimension_contributions.png'),
        os.path.join(output_dir, 'dimension_radar.png'),
        os.path.join(output_dir, 'question_timeline.png')
    ]
    
    return visualization_files

# Main Execution Function
def assess_question_difficulty(input_file, output_dir):
    """Main function to assess question difficulty."""
    # Initialize
    client = initialize_anthropic_client()
    os.makedirs(output_dir, exist_ok=True)
    
    # Load questions
    questions = load_questions_from_json(input_file)
    
    # Process each question
    all_results = []
    invalid_questions = []
    
    for i, question in enumerate(tqdm(questions, desc="Assessing questions")):
        # Validate question data
        is_valid, error_msg = validate_question_data(question)
        if not is_valid:
            print(f"Skipping invalid question at index {i}: {error_msg}")
            invalid_questions.append({
                "index": i,
                "question": question,
                "error": error_msg
            })
            continue
        
        try:
            # Evaluate each dimension
            knowledge_score, knowledge_details = aggregate_knowledge_requirements(question, client)
            cognitive_score, cognitive_details = aggregate_cognitive_complexity(question, client)
            context_score, context_details = aggregate_context_and_ambiguity(question, client)
            structure_score, structure_details = aggregate_question_structure(question, client)
            linguistic_score, linguistic_details = aggregate_linguistic_factors(question, client)
            objectives_score, objectives_details = aggregate_learning_objectives(question, client)
            
            # Combine dimension scores
            dimension_scores = {
                "knowledge_requirements": knowledge_score,
                "cognitive_complexity": cognitive_score,
                "context_and_ambiguity": context_score,
                "question_structure": structure_score,
                "linguistic_factors": linguistic_score,
                "learning_objectives": objectives_score
            }
            
            # Calculate overall score
            overall_score = calculate_overall_difficulty(dimension_scores)
            difficulty_level = determine_difficulty_level(overall_score)
            
            # Store results
            question_result = {
                "question_text": question["question"],
                "dimension_scores": dimension_scores,
                "overall_score": overall_score,
                "difficulty_level": difficulty_level,
                "details": {
                    "knowledge_requirements": knowledge_details,
                    "cognitive_complexity": cognitive_details,
                    "context_and_ambiguity": context_details,
                    "question_structure": structure_details,
                    "linguistic_factors": linguistic_details,
                    "learning_objectives": objectives_details
                }
            }
            all_results.append(question_result)
        except Exception as e:
            print(f"Error processing question at index {i}: {str(e)}")
            invalid_questions.append({
                "index": i,
                "question": question,
                "error": str(e)
            })
    
    # If all questions were invalid, raise an error
    if len(all_results) == 0:
        raise ValueError("No valid questions could be processed. Check the input data format.")
    
    # Calculate overall test difficulty
    overall_scores = [r["overall_score"] for r in all_results]
    avg_difficulty = statistics.mean(overall_scores)
    overall_difficulty_level = determine_difficulty_level(avg_difficulty)
    
    # Prepare final results
    final_results = {
        "questions": all_results,
        "overall_test_difficulty": {
            "score": avg_difficulty,
            "level": overall_difficulty_level,
            "dimension_averages": {
                "knowledge_requirements": statistics.mean([r["dimension_scores"]["knowledge_requirements"] for r in all_results]),
                "cognitive_complexity": statistics.mean([r["dimension_scores"]["cognitive_complexity"] for r in all_results]),
                "context_and_ambiguity": statistics.mean([r["dimension_scores"]["context_and_ambiguity"] for r in all_results]),
                "question_structure": statistics.mean([r["dimension_scores"]["question_structure"] for r in all_results]),
                "linguistic_factors": statistics.mean([r["dimension_scores"]["linguistic_factors"] for r in all_results]),
                "learning_objectives": statistics.mean([r["dimension_scores"]["learning_objectives"] for r in all_results])
            }
        },
        "invalid_questions": invalid_questions,  # Add info about invalid questions
        "processing_summary": {
            "total_questions": len(questions),
            "successful": len(all_results),
            "failed": len(invalid_questions)
        }
    }
    
    # Save and visualize results
    save_results_to_json(final_results, f"{output_dir}/difficulty_assessment.json")
    
    # Create visualizations
    visualization_files = create_difficulty_visualizations(final_results, output_dir)
    
    # Generate and save report in multiple formats
    try:
        # Generate reports in different formats
        md_report = generate_difficulty_report(all_results, final_results["overall_test_difficulty"], 
                                                output_format="markdown")
        html_report = generate_difficulty_report(all_results, final_results["overall_test_difficulty"], 
                                                 output_format="html")
        text_report = generate_difficulty_report(all_results, final_results["overall_test_difficulty"], 
                                                 output_format="text")
        
        # Save each format to appropriate files
        with open(f"{output_dir}/difficulty_report.md", 'w', encoding='utf-8') as f:
            f.write(md_report)
            
        with open(f"{output_dir}/difficulty_report.html", 'w', encoding='utf-8') as f:
            f.write(html_report)
            
        with open(f"{output_dir}/difficulty_report.txt", 'w', encoding='utf-8') as f:
            f.write(text_report)
            
        print(f"Difficulty reports saved to {output_dir}/difficulty_report.[md|html|txt]")
        
    except Exception as e:
        print(f"Warning: Failed to generate or save reports: {str(e)}")
    
    return final_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Assess the difficulty of questions in a JSON file")
    parser.add_argument("input_file", help="Path to the input JSON file containing questions")
    parser.add_argument("--output_dir", default="results", help="Directory to save results")
    args = parser.parse_args()
    
    # Process questions in parallel
    results = parallel_assess_difficulty(
        questions=questions,
        client=client,
        max_workers=5,  # Adjust based on your system capabilities
        show_progress=True
    )
    
    print(f"\nOverall Test Difficulty: {results['overall_test_difficulty']['level']} ({results['overall_test_difficulty']['score']:.2f})")
