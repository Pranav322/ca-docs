from openai import AzureOpenAI
from typing import List, Dict, Any
import logging
import numpy as np
import time
from config import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, AZURE_OPENAI_VERSION, AZURE_EMBEDDINGS_DEPLOYMENT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AzureEmbeddings:
    def __init__(self):
        # Configure Azure OpenAI client
        self.client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_KEY,
            api_version=AZURE_OPENAI_VERSION
        )
        
        self.deployment_name = AZURE_EMBEDDINGS_DEPLOYMENT
        self.max_tokens = 8191  # Max tokens for ada-002
        self.batch_size = 16    # Process embeddings in batches
        
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        try:
            # Truncate text if too long
            truncated_text = self._truncate_text(text)
            
            response = self.client.embeddings.create(
                input=truncated_text,
                model=self.deployment_name
            )
            
            embedding = response.data[0].embedding
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            raise
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts in batches"""
        try:
            all_embeddings = []
            
            # Process in batches to avoid rate limits
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                
                # Truncate texts in batch
                truncated_batch = [self._truncate_text(text) for text in batch]
                
                # Get embeddings for batch
                response = self.client.embeddings.create(
                    input=truncated_batch,
                    model=self.deployment_name
                )
                
                # Extract embeddings
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # Add small delay to respect rate limits
                if i + self.batch_size < len(texts):
                    time.sleep(0.1)
                
                logger.info(f"Processed embeddings batch {i//self.batch_size + 1}/{(len(texts)-1)//self.batch_size + 1}")
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Failed to get batch embeddings: {e}")
            raise
    
    def _truncate_text(self, text: str) -> str:
        """Truncate text to fit within token limits"""
        # Simple approximation: 4 characters ≈ 1 token
        max_chars = self.max_tokens * 4
        
        if len(text) <= max_chars:
            return text
        
        # Truncate and add indication
        truncated = text[:max_chars-50] + "\n[Text truncated for embedding...]"
        logger.warning(f"Text truncated from {len(text)} to {len(truncated)} characters")
        
        return truncated
    
    def get_table_embedding(self, table_text: str) -> List[float]:
        """Get embedding specifically optimized for table content"""
        try:
            # Add table-specific prefix to improve embedding quality
            enhanced_text = f"Financial/Accounting Table Data: {table_text}"
            
            return self.get_embedding(enhanced_text)
            
        except Exception as e:
            logger.error(f"Failed to get table embedding: {e}")
            raise
    
    def get_query_embedding(self, query: str, context_type: str = "general") -> List[float]:
        """Get embedding for user queries with context awareness"""
        try:
            # Enhance query based on context type
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
        """Calculate cosine similarity between two embeddings"""
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Failed to calculate cosine similarity: {e}")
            return 0.0
    
    def create_enhanced_embedding_text(self, content: str, metadata: Dict[str, Any]) -> str:
        """Create enhanced text for embedding that includes metadata context"""
        try:
            # Build context from metadata
            context_parts = []
            
            if metadata.get('level'):
                context_parts.append(f"CA {metadata['level']} level")
            if metadata.get('paper'):
                context_parts.append(f"Paper: {metadata['paper']}")
            if metadata.get('module'):
                context_parts.append(f"Module: {metadata['module']}")
            if metadata.get('chapter'):
                context_parts.append(f"Chapter: {metadata['chapter']}")
            if metadata.get('unit'):
                context_parts.append(f"Unit: {metadata['unit']}")
            
            # Add content type context
            content_type = metadata.get('content_type', 'text')
            if content_type == 'table':
                context_parts.append("Table/Financial Data")
            elif content_type == 'formula':
                context_parts.append("Formula/Calculation")
            elif content_type == 'example':
                context_parts.append("Example/Case Study")
            
            context_text = " | ".join(context_parts) if context_parts else ""
            
            # Combine context and content
            if context_text:
                enhanced_text = f"Context: {context_text} | Content: {content}"
            else:
                enhanced_text = content
            
            return enhanced_text
            
        except Exception as e:
            logger.error(f"Failed to create enhanced embedding text: {e}")
            return content

class EmbeddingManager:
    def __init__(self):
        self.azure_embeddings = AzureEmbeddings()
    
    def process_document_chunks(self, chunks: List[Dict[str, Any]], 
                              metadata: Dict[str, str]) -> List[Dict[str, Any]]:
        """Process document chunks and add embeddings"""
        try:
            processed_chunks = []
            
            # Prepare texts for batch embedding
            texts_for_embedding = []
            
            for chunk in chunks:
                # Create enhanced text with metadata context
                enhanced_text = self.azure_embeddings.create_enhanced_embedding_text(
                    chunk['content'], 
                    {**metadata, 'content_type': chunk.get('content_type', 'text')}
                )
                texts_for_embedding.append(enhanced_text)
            
            # Get embeddings in batch
            logger.info(f"Getting embeddings for {len(chunks)} document chunks...")
            embeddings = self.azure_embeddings.get_embeddings_batch(texts_for_embedding)
            
            # Combine chunks with embeddings
            for i, chunk in enumerate(chunks):
                processed_chunk = chunk.copy()
                processed_chunk['embedding'] = embeddings[i]
                processed_chunk['metadata'] = metadata
                processed_chunks.append(processed_chunk)
            
            logger.info(f"Successfully processed {len(processed_chunks)} chunks with embeddings")
            return processed_chunks
            
        except Exception as e:
            logger.error(f"Failed to process document chunks: {e}")
            raise
    
    def process_tables(self, tables: List[Dict[str, Any]], 
                      metadata: Dict[str, str]) -> List[Dict[str, Any]]:
        """Process tables and add embeddings"""
        try:
            from table_processor import TableProcessor
            
            table_processor = TableProcessor()
            processed_tables = []
            
            # Prepare table texts for embedding
            table_texts = []
            
            for table in tables:
                # Create comprehensive table embedding text
                table_embedding_text = table_processor.create_table_embedding_text(table, metadata)
                table_texts.append(table_embedding_text)
            
            # Get embeddings for tables
            logger.info(f"Getting embeddings for {len(tables)} tables...")
            embeddings = self.azure_embeddings.get_embeddings_batch(table_texts)
            
            # Combine tables with embeddings
            for i, table in enumerate(tables):
                processed_table = table.copy()
                processed_table['embedding'] = embeddings[i]
                processed_table['metadata'] = metadata
                processed_table['embedding_text'] = table_texts[i]
                processed_tables.append(processed_table)
            
            logger.info(f"Successfully processed {len(processed_tables)} tables with embeddings")
            return processed_tables
            
        except Exception as e:
            logger.error(f"Failed to process tables: {e}")
            raise
    
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
