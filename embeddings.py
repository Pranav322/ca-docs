import asyncio
import aiohttp
import json
from openai import AzureOpenAI
from typing import List, Dict, Any, Optional
import logging
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
import threading
from config import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, AZURE_OPENAI_VERSION, AZURE_EMBEDDINGS_DEPLOYMENT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedAzureEmbeddings:
    def __init__(self, max_concurrent_requests: int = 10):
        # Configure Azure OpenAI client
        self.client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_KEY,
            api_version=AZURE_OPENAI_VERSION
        )
        
        self.deployment_name = AZURE_EMBEDDINGS_DEPLOYMENT
        self.max_tokens = 8191  # Max tokens for ada-002
        self.batch_size = 100    # Increased batch size for better throughput
        self.max_concurrent_requests = max_concurrent_requests
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        
        # Rate limiting
        self.requests_per_minute = 3000  # Azure OpenAI rate limit
        self.request_times = []
        self.rate_limit_lock = threading.Lock()
        
    async def get_embeddings_batch_async(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts with async processing and rate limiting"""
        try:
            if not texts:
                return []
            
            # Pre-process texts (truncate, etc.)
            processed_texts = [self._truncate_text(text) for text in texts]
            
            # Process in batches with async concurrency
            all_embeddings = []
            
            # Create batches
            batches = [processed_texts[i:i + self.batch_size] for i in range(0, len(processed_texts), self.batch_size)]
            
            # Process batches concurrently with semaphore
            tasks = []
            for batch in batches:
                task = self._process_batch_with_rate_limit(batch)
                tasks.append(task)
            
            # Wait for all batches to complete
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine results
            for result in batch_results:
                if isinstance(result, list):
                    all_embeddings.extend(result)
                else:
                    logger.error(f"Batch processing failed: {result}")
                    # Add empty embeddings for failed batch
                    batch_size = len(batches[0]) if batches else 0
                    all_embeddings.extend([[] for _ in range(batch_size)])
            
            logger.info(f"Successfully processed {len(all_embeddings)} embeddings")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Failed to get batch embeddings: {e}")
            raise
    
    async def _process_batch_with_rate_limit(self, batch: List[str]) -> List[List[float]]:
        """Process a single batch with rate limiting"""
        async with self.semaphore:
            # Check rate limit
            await self._check_rate_limit()
            
            # Process batch
            return await self._process_batch(batch)
    
    async def _check_rate_limit(self):
        """Check and enforce rate limits"""
        with self.rate_limit_lock:
            now = time.time()
            # Remove requests older than 1 minute
            self.request_times = [t for t in self.request_times if now - t < 60]
            
            # If we're at the rate limit, wait
            if len(self.request_times) >= self.requests_per_minute:
                sleep_time = 60 - (now - self.request_times[0]) + 1
                if sleep_time > 0:
                    logger.info(f"Rate limit reached, sleeping for {sleep_time:.2f}s")
                    await asyncio.sleep(sleep_time)
                    # Clean up old times after sleep
                    now = time.time()
                    self.request_times = [t for t in self.request_times if now - t < 60]
            
            # Record this request
            self.request_times.append(now)
    
    async def _process_batch(self, batch: List[str]) -> List[List[float]]:
        """Process a single batch of texts"""
        try:
            # Run the synchronous OpenAI call in a thread pool
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                embeddings = await loop.run_in_executor(
                    executor, 
                    self._get_embeddings_sync, 
                    batch
                )
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Return empty embeddings for failed batch
            return [[] for _ in batch]
    
    def _get_embeddings_sync(self, batch: List[str]) -> List[List[float]]:
        """Synchronous call to get embeddings for a batch"""
        try:
            response = self.client.embeddings.create(
                input=batch,
                model=self.deployment_name
            )
            
            return [item.embedding for item in response.data]
            
        except Exception as e:
            logger.error(f"Sync embedding call failed: {e}")
            raise
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text (sync wrapper)"""
        try:
            truncated_text = self._truncate_text(text)
            
            response = self.client.embeddings.create(
                input=truncated_text,
                model=self.deployment_name
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            raise
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Synchronous wrapper for batch embeddings"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.get_embeddings_batch_async(texts))
        finally:
            loop.close()
    
    def _truncate_text(self, text: str) -> str:
        """Truncate text to fit within token limits - optimized"""
        # Simple approximation: 4 characters ≈ 1 token
        max_chars = self.max_tokens * 4
        
        if len(text) <= max_chars:
            return text
        
        # Truncate and add indication
        truncated = text[:max_chars-50] + "\n[Text truncated for embedding...]"
        return truncated
    
    def get_table_embedding(self, table_text: str) -> List[float]:
        """Get embedding specifically optimized for table content"""
        try:
            enhanced_text = f"Financial/Accounting Table Data: {table_text}"
            return self.get_embedding(enhanced_text)
        except Exception as e:
            logger.error(f"Failed to get table embedding: {e}")
            raise
    
    def get_query_embedding(self, query: str, context_type: str = "general") -> List[float]:
        """Get embedding for user queries with context awareness"""
        try:
            if context_type == "table":
                enhanced_query = f"Search for table/financial data: {query}"
            elif context_type == "ca_syllabus":
                enhanced_query = f"CA syllabus question: {query}"
            else:
                enhanced_query = query
            
            return self.get_embedding(enhanced_query)
            
        except Exception as e:
            logger.error(f"Failed to get query embedding: {e}")
            raise
    
    def cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings - optimized"""
        try:
            # Convert to numpy arrays for faster computation
            vec1 = np.array(embedding1, dtype=np.float32)
            vec2 = np.array(embedding2, dtype=np.float32)
            
            # Calculate cosine similarity using numpy operations
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
            
        except Exception as e:
            logger.error(f"Failed to calculate cosine similarity: {e}")
            return 0.0
    
    def create_enhanced_embedding_text(self, content: str, metadata: Dict[str, Any]) -> str:
        """Create enhanced text for embedding that includes metadata context - optimized"""
        try:
            # Build context from metadata more efficiently
            context_parts = []
            
            # Use dict.get() for cleaner code
            level = metadata.get('level')
            paper = metadata.get('paper')
            module = metadata.get('module')
            chapter = metadata.get('chapter')
            unit = metadata.get('unit')
            
            if level:
                context_parts.append(f"CA {level} level")
            if paper:
                context_parts.append(f"Paper: {paper}")
            if module:
                context_parts.append(f"Module: {module}")
            if chapter:
                context_parts.append(f"Chapter: {chapter}")
            if unit:
                context_parts.append(f"Unit: {unit}")
            
            # Add content type context
            content_type = metadata.get('content_type', 'text')
            if content_type == 'table':
                context_parts.append("Table/Financial Data")
            elif content_type == 'formula':
                context_parts.append("Formula/Calculation")
            elif content_type == 'example':
                context_parts.append("Example/Case Study")
            
            # Combine context and content efficiently
            if context_parts:
                enhanced_text = f"Context: {' | '.join(context_parts)} | Content: {content}"
            else:
                enhanced_text = content
            
            return enhanced_text
            
        except Exception as e:
            logger.error(f"Failed to create enhanced embedding text: {e}")
            return content

class OptimizedEmbeddingManager:
    def __init__(self, max_concurrent_requests: int = 10):
        self.azure_embeddings = OptimizedAzureEmbeddings(max_concurrent_requests)
        self.embedding_cache = {}  # Simple in-memory cache
        self.cache_hits = 0
        self.cache_misses = 0
    
    async def process_document_chunks_async(self, chunks: List[Dict[str, Any]], 
                                          metadata: Dict[str, str]) -> List[Dict[str, Any]]:
        """Process document chunks with async embedding generation"""
        try:
            if not chunks:
                return []
            
            processed_chunks = []
            
            # Prepare texts for batch embedding
            texts_for_embedding = []
            cache_keys = []
            
            for chunk in chunks:
                # Create enhanced text with metadata context
                enhanced_text = self.azure_embeddings.create_enhanced_embedding_text(
                    chunk['content'], 
                    {**metadata, 'content_type': chunk.get('content_type', 'text')}
                )
                
                # Check cache first
                cache_key = hash(enhanced_text)
                cache_keys.append(cache_key)
                
                if cache_key in self.embedding_cache:
                    self.cache_hits += 1
                    texts_for_embedding.append(None)  # Placeholder for cached embedding
                else:
                    self.cache_misses += 1
                    texts_for_embedding.append(enhanced_text)
            
            # Get embeddings in batch (including cached ones)
            embeddings = await self._get_embeddings_with_cache(texts_for_embedding, cache_keys)
            
            # Combine chunks with embeddings
            for i, chunk in enumerate(chunks):
                processed_chunk = chunk.copy()
                processed_chunk['embedding'] = embeddings[i]
                processed_chunk['metadata'] = metadata
                processed_chunks.append(processed_chunk)
            
            logger.info(f"Successfully processed {len(processed_chunks)} chunks with embeddings (cache hits: {self.cache_hits}, misses: {self.cache_misses})")
            return processed_chunks
            
        except Exception as e:
            logger.error(f"Failed to process document chunks: {e}")
            raise
    
    async def _get_embeddings_with_cache(self, texts: List[str], cache_keys: List[int]) -> List[List[float]]:
        """Get embeddings with cache support"""
        embeddings = []
        texts_to_process = []
        indices_to_process = []
        
        # Separate cached and non-cached embeddings
        for i, (text, cache_key) in enumerate(zip(texts, cache_keys)):
            if text is None:  # Cached
                embeddings.append(self.embedding_cache[cache_key])
            else:
                embeddings.append(None)  # Placeholder
                texts_to_process.append(text)
                indices_to_process.append(i)
        
        # Get embeddings for non-cached texts
        if texts_to_process:
            new_embeddings = await self.azure_embeddings.get_embeddings_batch_async(texts_to_process)
            
            # Update embeddings and cache
            for idx, embedding in zip(indices_to_process, new_embeddings):
                embeddings[idx] = embedding
                # Cache the embedding
                self.embedding_cache[cache_keys[idx]] = embedding
        
        return embeddings
    
    def process_document_chunks(self, chunks: List[Dict[str, Any]], 
                              metadata: Dict[str, str]) -> List[Dict[str, Any]]:
        """Synchronous wrapper for async processing"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.process_document_chunks_async(chunks, metadata))
        finally:
            loop.close()
    
    async def process_tables_async(self, tables: List[Dict[str, Any]], 
                                 metadata: Dict[str, str]) -> List[Dict[str, Any]]:
        """Process tables with async embedding generation"""
        try:
            from table_processor import TableProcessor
            
            table_processor = TableProcessor()
            processed_tables = []
            
            # Prepare table texts for embedding
            table_texts = []
            cache_keys = []
            
            for table in tables:
                # Create comprehensive table embedding text
                table_embedding_text = table_processor.create_table_embedding_text(table, metadata)
                
                # Check cache
                cache_key = hash(table_embedding_text)
                cache_keys.append(cache_key)
                
                if cache_key in self.embedding_cache:
                    self.cache_hits += 1
                    table_texts.append(None)  # Placeholder for cached embedding
                else:
                    self.cache_misses += 1
                    table_texts.append(table_embedding_text)
            
            # Get embeddings for tables
            logger.info(f"Getting embeddings for {len(tables)} tables...")
            embeddings = await self._get_embeddings_with_cache(table_texts, cache_keys)
            
            # Combine tables with embeddings
            for i, table in enumerate(tables):
                processed_table = table.copy()
                processed_table['embedding'] = embeddings[i]
                processed_table['metadata'] = metadata
                processed_table['embedding_text'] = table_texts[i] if table_texts[i] else f"Cached embedding for table {i}"
                processed_tables.append(processed_table)
            
            logger.info(f"Successfully processed {len(processed_tables)} tables with embeddings")
            return processed_tables
            
        except Exception as e:
            logger.error(f"Failed to process tables: {e}")
            raise
    
    def process_tables(self, tables: List[Dict[str, Any]], 
                      metadata: Dict[str, str]) -> List[Dict[str, Any]]:
        """Synchronous wrapper for async table processing"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.process_tables_async(tables, metadata))
        finally:
            loop.close()
    
    def get_query_embedding_with_filters(self, query: str, level: str = None, 
                                       paper: str = None, content_type: str = "general") -> List[float]:
        """Get query embedding with context from filters"""
        try:
            # Enhance query with filter context
            enhanced_parts = [query]
            
            if level:
                enhanced_parts.append(f"CA {level} level")
            if paper:
                enhanced_parts.append(f"Paper {paper}")
            
            enhanced_query = " ".join(enhanced_parts)
            
            return self.azure_embeddings.get_query_embedding(enhanced_query, content_type)
            
        except Exception as e:
            logger.error(f"Failed to get query embedding: {e}")
            raise
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get embedding cache statistics"""
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_size': len(self.embedding_cache),
            'hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
        }
    
    def clear_cache(self):
        """Clear the embedding cache"""
        self.embedding_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("Embedding cache cleared")

# Aliases for backward compatibility
AzureEmbeddings = OptimizedAzureEmbeddings
EmbeddingManager = OptimizedEmbeddingManager



