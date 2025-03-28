"""
Cache manager for the Question Difficulty Framework.

This module provides functionality for caching LLM responses to avoid
redundant API calls and speed up question assessments.
"""

import os
import json
import hashlib
import time
import logging
from functools import wraps
from typing import Dict, Any, Optional, Callable

# Get logger
logger = logging.getLogger("difficulty_framework")

class DifficultyCache:
    """
    Manages caching of LLM responses for the Question Difficulty Framework.
    """
    
    def __init__(self, cache_dir: str = None, ttl_days: int = 30, enabled: bool = True):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir: Directory to store cache files (default: 'cache' in the current directory)
            ttl_days: Time-to-live for cache entries in days (default: 30)
            enabled: Whether caching is enabled (default: True)
        """
        self.enabled = enabled
        
        if not enabled:
            logger.info("Caching is disabled")
            return
        
        # Set up cache directory
        if cache_dir is None:
            # Create cache in the current working directory by default
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
        
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_days * 24 * 60 * 60  # Convert days to seconds
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info(f"Cache initialized at {self.cache_dir} with TTL of {ttl_days} days")
        
        # Stats
        self.hits = 0
        self.misses = 0
        self.errors = 0
    
    def generate_key(self, prompt: str, model: str = None) -> str:
        """
        Generate a unique cache key for a prompt and model.
        
        Args:
            prompt: The LLM prompt
            model: The LLM model name (optional)
            
        Returns:
            A unique cache key as a hexadecimal string
        """
        # Create a hash of the prompt and model
        key_content = prompt
        if model:
            key_content += f"|{model}"
        
        # Generate MD5 hash
        return hashlib.md5(key_content.encode('utf-8')).hexdigest()
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a cached response.
        
        Args:
            key: The cache key
            
        Returns:
            The cached response or None if not found
        """
        if not self.enabled:
            return None
        
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        
        # Check if cache file exists
        if not os.path.exists(cache_file):
            self.misses += 1
            return None
        
        try:
            # Check if cache has expired
            mod_time = os.path.getmtime(cache_file)
            if time.time() - mod_time > self.ttl_seconds:
                # Cache entry has expired
                logger.debug(f"Cache entry expired: {key}")
                os.remove(cache_file)
                self.misses += 1
                return None
            
            # Read and return cached data
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            
            logger.debug(f"Cache hit: {key}")
            self.hits += 1
            return cached_data
            
        except Exception as e:
            logger.warning(f"Error reading cache: {e}")
            self.errors += 1
            return None
    
    def set(self, key: str, data: Any) -> bool:
        """
        Store a response in the cache.
        
        Args:
            key: The cache key
            data: The data to cache
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False
        
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        
        try:
            # Write data to cache file
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"Cached: {key}")
            return True
            
        except Exception as e:
            logger.warning(f"Error writing to cache: {e}")
            self.errors += 1
            return False
    
    def clear_expired(self) -> int:
        """
        Clear expired cache entries.
        
        Returns:
            Number of cleared entries
        """
        if not self.enabled:
            return 0
        
        cleared = 0
        now = time.time()
        
        for filename in os.listdir(self.cache_dir):
            if not filename.endswith('.json'):
                continue
                
            filepath = os.path.join(self.cache_dir, filename)
            
            try:
                mod_time = os.path.getmtime(filepath)
                if now - mod_time > self.ttl_seconds:
                    os.remove(filepath)
                    cleared += 1
            except Exception as e:
                logger.warning(f"Error clearing cache entry {filename}: {e}")
        
        logger.info(f"Cleared {cleared} expired cache entries")
        return cleared
    
    def clear_all(self) -> int:
        """
        Clear all cache entries.
        
        Returns:
            Number of cleared entries
        """
        if not self.enabled:
            return 0
        
        cleared = 0
        
        for filename in os.listdir(self.cache_dir):
            if not filename.endswith('.json'):
                continue
                
            filepath = os.path.join(self.cache_dir, filename)
            
            try:
                os.remove(filepath)
                cleared += 1
            except Exception as e:
                logger.warning(f"Error clearing cache entry {filename}: {e}")
        
        logger.info(f"Cleared all {cleared} cache entries")
        return cleared
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total": total,
            "hit_rate": hit_rate,
            "errors": self.errors,
            "enabled": self.enabled,
            "ttl_days": self.ttl_seconds / (24 * 60 * 60)
        }

# Global cache instance
_cache = None

def get_cache(cache_dir: str = None, ttl_days: int = 30, enabled: bool = True) -> DifficultyCache:
    """
    Get the global cache instance, creating it if necessary.
    
    Args:
        cache_dir: Directory to store cache files
        ttl_days: Time-to-live for cache entries in days
        enabled: Whether caching is enabled
        
    Returns:
        The global cache instance
    """
    global _cache
    
    if _cache is None:
        _cache = DifficultyCache(cache_dir, ttl_days, enabled)
    
    return _cache

def cache_llm_response(func: Callable) -> Callable:
    """
    Decorator to cache LLM responses.
    
    Args:
        func: The function to decorate
        
    Returns:
        Decorated function with caching
    """
    @wraps(func)
    def wrapper(client, prompt, *args, **kwargs):
        # Skip caching if specified
        if kwargs.pop('skip_cache', False):
            return func(client, prompt, *args, **kwargs)
        
        # Get cache instance
        cache = get_cache()
        
        if not cache.enabled:
            return func(client, prompt, *args, **kwargs)
        
        # Generate cache key
        model = kwargs.get('model', 'default')
        cache_key = cache.generate_key(prompt, model)
        
        # Try to get from cache
        cached_result = cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Call the function
        result = func(client, prompt, *args, **kwargs)
        
        # Cache the result
        cache.set(cache_key, result)
        
        return result
    
    return wrapper
