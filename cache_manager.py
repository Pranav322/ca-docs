import redis
import json
import pickle
import hashlib
import time
import logging
from typing import Any, Dict, List, Optional, Union
from functools import wraps
import asyncio
from datetime import datetime, timedelta
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CacheManager:
    """Comprehensive caching system for CA RAG application"""
    
    def __init__(self, redis_url: str = None, default_ttl: int = 3600):
        self.redis_client = None
        self.default_ttl = default_ttl
        self.memory_cache = {}  # Fallback in-memory cache
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0
        }
        
        # Try to connect to Redis
        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url, decode_responses=False)
                self.redis_client.ping()  # Test connection
                logger.info("Connected to Redis cache")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}. Using in-memory cache.")
                self.redis_client = None
        
        # Cache key prefixes for organization
        self.key_prefixes = {
            'embeddings': 'emb:',
            'search_results': 'search:',
            'file_metadata': 'file:',
            'pdf_content': 'pdf:',
            'table_content': 'table:',
            'query_results': 'query:',
            'curriculum_data': 'curriculum:'
        }
    
    def _get_cache_key(self, prefix: str, key: str) -> str:
        """Generate cache key with prefix"""
        return f"{self.key_prefixes.get(prefix, '')}{key}"
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for storage"""
        try:
            if isinstance(value, (str, int, float, bool)):
                return json.dumps(value).encode('utf-8')
            else:
                return pickle.dumps(value)
        except Exception as e:
            logger.error(f"Serialization failed: {e}")
            return pickle.dumps(value)
    
    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value from storage"""
        try:
            # Try JSON first for simple types
            try:
                return json.loads(data.decode('utf-8'))
            except:
                pass
            
            # Fall back to pickle
            return pickle.loads(data)
        except Exception as e:
            logger.error(f"Deserialization failed: {e}")
            return None
    
    def get(self, prefix: str, key: str) -> Optional[Any]:
        """Get value from cache"""
        cache_key = self._get_cache_key(prefix, key)
        
        try:
            if self.redis_client:
                data = self.redis_client.get(cache_key)
                if data is not None:
                    self.cache_stats['hits'] += 1
                    return self._deserialize_value(data)
                else:
                    self.cache_stats['misses'] += 1
                    return None
            else:
                # Use in-memory cache
                if cache_key in self.memory_cache:
                    item = self.memory_cache[cache_key]
                    # Check TTL
                    if item['expires'] > time.time():
                        self.cache_stats['hits'] += 1
                        return item['value']
                    else:
                        del self.memory_cache[cache_key]
                
                self.cache_stats['misses'] += 1
                return None
                
        except Exception as e:
            logger.error(f"Cache get failed: {e}")
            self.cache_stats['misses'] += 1
            return None
    
    def set(self, prefix: str, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in cache"""
        cache_key = self._get_cache_key(prefix, key)
        ttl = ttl or self.default_ttl
        
        try:
            serialized_value = self._serialize_value(value)
            
            if self.redis_client:
                success = self.redis_client.setex(cache_key, ttl, serialized_value)
                self.cache_stats['sets'] += 1
                return bool(success)
            else:
                # Use in-memory cache
                self.memory_cache[cache_key] = {
                    'value': value,
                    'expires': time.time() + ttl
                }
                self.cache_stats['sets'] += 1
                return True
                
        except Exception as e:
            logger.error(f"Cache set failed: {e}")
            return False
    
    def delete(self, prefix: str, key: str) -> bool:
        """Delete value from cache"""
        cache_key = self._get_cache_key(prefix, key)
        
        try:
            if self.redis_client:
                success = self.redis_client.delete(cache_key)
                self.cache_stats['deletes'] += 1
                return bool(success)
            else:
                if cache_key in self.memory_cache:
                    del self.memory_cache[cache_key]
                    self.cache_stats['deletes'] += 1
                return True
                
        except Exception as e:
            logger.error(f"Cache delete failed: {e}")
            return False
    
    def exists(self, prefix: str, key: str) -> bool:
        """Check if key exists in cache"""
        cache_key = self._get_cache_key(prefix, key)
        
        try:
            if self.redis_client:
                return bool(self.redis_client.exists(cache_key))
            else:
                if cache_key in self.memory_cache:
                    item = self.memory_cache[cache_key]
                    if item['expires'] > time.time():
                        return True
                    else:
                        del self.memory_cache[cache_key]
                return False
                
        except Exception as e:
            logger.error(f"Cache exists check failed: {e}")
            return False
    
    def clear_prefix(self, prefix: str) -> int:
        """Clear all keys with a specific prefix"""
        try:
            if self.redis_client:
                pattern = f"{self.key_prefixes.get(prefix, '')}*"
                keys = self.redis_client.keys(pattern)
                if keys:
                    deleted = self.redis_client.delete(*keys)
                    return deleted
                return 0
            else:
                # Clear from in-memory cache
                pattern = f"{self.key_prefixes.get(prefix, '')}"
                keys_to_delete = [k for k in self.memory_cache.keys() if k.startswith(pattern)]
                for key in keys_to_delete:
                    del self.memory_cache[key]
                return len(keys_to_delete)
                
        except Exception as e:
            logger.error(f"Cache clear prefix failed: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = (self.cache_stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        stats = {
            'hits': self.cache_stats['hits'],
            'misses': self.cache_stats['misses'],
            'sets': self.cache_stats['sets'],
            'deletes': self.cache_stats['deletes'],
            'hit_rate': round(hit_rate, 2),
            'total_requests': total_requests,
            'cache_type': 'Redis' if self.redis_client else 'Memory',
            'memory_cache_size': len(self.memory_cache) if not self.redis_client else 0
        }
        
        # Add Redis-specific stats if available
        if self.redis_client:
            try:
                info = self.redis_client.info()
                stats.update({
                    'redis_memory_used': info.get('used_memory_human', 'Unknown'),
                    'redis_keys': info.get('db0', {}).get('keys', 0)
                })
            except:
                pass
        
        return stats
    
    def reset_stats(self):
        """Reset cache statistics"""
        self.cache_stats = {'hits': 0, 'misses': 0, 'sets': 0, 'deletes': 0}

# Specific cache managers for different components
class EmbeddingCache:
    """Cache manager specifically for embeddings"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
        self.prefix = 'embeddings'
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding for text"""
        key = hashlib.md5(text.encode()).hexdigest()
        return self.cache.get(self.prefix, key)
    
    def set_embedding(self, text: str, embedding: List[float], ttl: int = 7200) -> bool:
        """Cache embedding for text"""
        key = hashlib.md5(text.encode()).hexdigest()
        return self.cache.set(self.prefix, key, embedding, ttl)
    
    def get_batch_embeddings(self, texts: List[str]) -> Dict[str, List[float]]:
        """Get multiple cached embeddings"""
        cached = {}
        uncached_indices = []
        
        for i, text in enumerate(texts):
            embedding = self.get_embedding(text)
            if embedding:
                cached[text] = embedding
            else:
                uncached_indices.append(i)
        
        return cached, uncached_indices
    
    def set_batch_embeddings(self, text_embedding_pairs: Dict[str, List[float]], ttl: int = 7200):
        """Cache multiple embeddings"""
        for text, embedding in text_embedding_pairs.items():
            self.set_embedding(text, embedding, ttl)

class SearchResultCache:
    """Cache manager for search results"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
        self.prefix = 'search_results'
    
    def get_search_result(self, query: str, filters: Dict[str, Any], content_type: str = 'documents') -> Optional[List[Dict]]:
        """Get cached search results"""
        key = self._generate_search_key(query, filters, content_type)
        return self.cache.get(self.prefix, key)
    
    def set_search_result(self, query: str, filters: Dict[str, Any], results: List[Dict], 
                         content_type: str = 'documents', ttl: int = 1800) -> bool:
        """Cache search results"""
        key = self._generate_search_key(query, filters, content_type)
        return self.cache.set(self.prefix, key, results, ttl)
    
    def _generate_search_key(self, query: str, filters: Dict[str, Any], content_type: str) -> str:
        """Generate cache key for search results"""
        key_data = {
            'query': query.lower().strip(),
            'filters': sorted(filters.items()) if filters else [],
            'content_type': content_type
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()

class FileContentCache:
    """Cache manager for file content"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
        self.pdf_prefix = 'pdf_content'
        self.table_prefix = 'table_content'
    
    def get_pdf_content(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get cached PDF content"""
        return self.cache.get(self.pdf_prefix, file_id)
    
    def set_pdf_content(self, file_id: str, content: Dict[str, Any], ttl: int = 3600) -> bool:
        """Cache PDF content"""
        return self.cache.set(self.pdf_prefix, file_id, content, ttl)
    
    def get_table_content(self, file_id: str, table_index: int) -> Optional[Dict[str, Any]]:
        """Get cached table content"""
        key = f"{file_id}_{table_index}"
        return self.cache.get(self.table_prefix, key)
    
    def set_table_content(self, file_id: str, table_index: int, content: Dict[str, Any], ttl: int = 3600) -> bool:
        """Cache table content"""
        key = f"{file_id}_{table_index}"
        return self.cache.set(self.table_prefix, key, content, ttl)

# Decorator for automatic caching
def cached(cache_manager: CacheManager, prefix: str, ttl: int = 3600, key_func=None):
    """Decorator to automatically cache function results"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation from args and kwargs
                key_data = {'args': args, 'kwargs': kwargs}
                cache_key = hashlib.md5(str(key_data).encode()).hexdigest()
            
            # Try to get from cache
            result = cache_manager.get(prefix, cache_key)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.set(prefix, cache_key, result, ttl)
            return result
        
        return wrapper
    return decorator

# Global cache manager instance
def get_cache_manager() -> CacheManager:
    """Get global cache manager instance"""
    redis_url = os.getenv('REDIS_URL')
    return CacheManager(redis_url=redis_url)

# Initialize global cache manager
cache_manager = get_cache_manager()
embedding_cache = EmbeddingCache(cache_manager)
search_cache = SearchResultCache(cache_manager)
file_cache = FileContentCache(cache_manager)


