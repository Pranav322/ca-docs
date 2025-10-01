import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor, execute_values
import numpy as np
import json
from typing import List, Dict, Any, Optional
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from config import DATABASE_URL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedVectorDatabase:
    def __init__(self, max_connections: int = 20, batch_size: int = 1000):
        self.connection_url = DATABASE_URL
        self.pool = None
        self.max_connections = max_connections
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.init_database()
    
    def get_connection(self):
        """Get database connection from the pool"""
        try:
            if self.pool is None:
                self.pool = pool.SimpleConnectionPool(
                    1, self.max_connections, 
                    dsn=self.connection_url, 
                    cursor_factory=RealDictCursor
                )
            conn = self.pool.getconn()
            return conn
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def init_database(self):
        """Initialize database with required tables and extensions"""
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            
            # Enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create documents table with optimized structure
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    file_id TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector(1536),
                    metadata JSONB,
                    chunk_index INTEGER,
                    level TEXT,
                    paper TEXT,
                    module TEXT,
                    chapter TEXT,
                    unit TEXT,
                    content_type TEXT DEFAULT 'text',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create tables table with optimized structure
            cur.execute("""
                CREATE TABLE IF NOT EXISTS tables (
                    id SERIAL PRIMARY KEY,
                    file_id TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    table_data JSONB NOT NULL,
                    table_html TEXT,
                    embedding vector(1536),
                    context_before TEXT,
                    context_after TEXT,
                    page_number INTEGER,
                    table_index INTEGER,
                    level TEXT,
                    paper TEXT,
                    module TEXT,
                    chapter TEXT,
                    unit TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create file_metadata table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS file_metadata (
                    id SERIAL PRIMARY KEY,
                    file_id TEXT UNIQUE NOT NULL,
                    file_name TEXT NOT NULL,
                    appwrite_file_id TEXT,
                    source_file TEXT,
                    level TEXT,
                    paper TEXT,
                    module TEXT,
                    chapter TEXT,
                    unit TEXT,
                    total_pages INTEGER,
                    processing_status TEXT DEFAULT 'pending',
                    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processed_date TIMESTAMP
                );
            """)
            
            # Add source_file column if it doesn't exist
            cur.execute("""
                ALTER TABLE file_metadata 
                ADD COLUMN IF NOT EXISTS source_file TEXT;
            """)
            
            # Create optimized indexes
            self._create_optimized_indexes(cur)
            
            conn.commit()
            cur.close()
            self.pool.putconn(conn)
            
            logger.info("Optimized database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def _create_optimized_indexes(self, cur):
        """Create optimized indexes for better performance"""
        try:
            # Vector similarity indexes
            cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_embedding ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_tables_embedding ON tables USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);")
            
            # Metadata indexes
            cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_metadata ON documents USING gin(metadata);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_tables_metadata ON tables USING gin(table_data);")
            
            # Hierarchical indexes
            cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_hierarchy ON documents(level, paper, module, chapter, unit);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_tables_hierarchy ON tables(level, paper, module, chapter, unit);")
            
            # File-based indexes
            cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_file_id ON documents(file_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_tables_file_id ON tables(file_id);")
            
            # Status and date indexes
            cur.execute("CREATE INDEX IF NOT EXISTS idx_file_metadata_status ON file_metadata(processing_status);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_file_metadata_upload_date ON file_metadata(upload_date);")
            
            logger.info("Optimized indexes created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")
            raise
    
    def store_document_chunks_batch(self, chunks_data: List[Dict[str, Any]]) -> List[int]:
        """Store multiple document chunks efficiently"""
        try:
            if not chunks_data:
                logger.warning("No chunks data provided to store")
                return []
            
            logger.info(f"Storing {len(chunks_data)} document chunks")
            
            # Use individual inserts in a transaction for reliability
            doc_ids = []
            conn = self.get_connection()
            cur = conn.cursor()
            
            try:
                for i, chunk in enumerate(chunks_data):
                    cur.execute("""
                        INSERT INTO documents (file_id, file_name, content, embedding, metadata, 
                                             chunk_index, level, paper, module, chapter, unit, content_type)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id;
                    """, (
                        chunk['file_id'],
                        chunk['file_name'],
                        chunk['content'],
                        chunk['embedding'],
                        json.dumps(chunk['metadata']),
                        chunk['chunk_index'],
                        chunk.get('level', ''),
                        chunk.get('paper', ''),
                        chunk.get('module', ''),
                        chunk.get('chapter', ''),
                        chunk.get('unit', ''),
                        chunk.get('content_type', 'text')
                    ))
                    
                    result = cur.fetchone()
                    if result:
                        doc_ids.append(result[0])
                    
                    # Commit every 100 inserts to avoid long transactions
                    if (i + 1) % 100 == 0:
                        conn.commit()
                        logger.info(f"Committed {i + 1} chunks")
                
                # Final commit
                conn.commit()
                
            except Exception as e:
                conn.rollback()
                raise e
            finally:
                cur.close()
                self.pool.putconn(conn)
            
            logger.info(f"Successfully stored {len(doc_ids)} document chunks")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Failed to store document chunks: {e}")
            logger.error(f"Chunks data length: {len(chunks_data) if chunks_data else 0}")
            if chunks_data:
                logger.error(f"First chunk keys: {list(chunks_data[0].keys())}")
            raise
    
    def store_document_chunk(self, file_id: str, file_name: str, content: str, 
                           embedding: List[float], metadata: Dict, chunk_index: int,
                           level: str = None, paper: str = None, module: str = None,
                           chapter: str = None, unit: str = None) -> int:
        """Store a single document chunk (legacy method for compatibility)"""
        chunks_data = [{
            'file_id': file_id,
            'file_name': file_name,
            'content': content,
            'embedding': embedding,
            'metadata': metadata,
            'chunk_index': chunk_index,
            'level': level,
            'paper': paper,
            'module': module,
            'chapter': chapter,
            'unit': unit
        }]
        
        ids = self.store_document_chunks_batch(chunks_data)
        return ids[0] if ids else None
    
    def store_tables_batch(self, tables_data: List[Dict[str, Any]]) -> List[int]:
        """Store multiple tables efficiently"""
        try:
            if not tables_data:
                logger.warning("No tables data provided to store")
                return []
            
            logger.info(f"Storing {len(tables_data)} tables")
            
            # Use individual inserts in a transaction for reliability
            table_ids = []
            conn = self.get_connection()
            cur = conn.cursor()
            
            try:
                for i, table in enumerate(tables_data):
                    cur.execute("""
                        INSERT INTO tables (file_id, file_name, table_data, table_html, embedding,
                                          context_before, context_after, page_number, table_index,
                                          level, paper, module, chapter, unit)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id;
                    """, (
                        table['file_id'],
                        table['file_name'],
                        json.dumps(table['table_data']),
                        table.get('table_html', ''),
                        table['embedding'],
                        table.get('context_before', ''),
                        table.get('context_after', ''),
                        table.get('page_number', 0),
                        table.get('table_index', 0),
                        table.get('level', ''),
                        table.get('paper', ''),
                        table.get('module', ''),
                        table.get('chapter', ''),
                        table.get('unit', '')
                    ))
                    
                    result = cur.fetchone()
                    if result:
                        table_ids.append(result[0])
                    
                    # Commit every 50 inserts to avoid long transactions
                    if (i + 1) % 50 == 0:
                        conn.commit()
                        logger.info(f"Committed {i + 1} tables")
                
                # Final commit
                conn.commit()
                
            except Exception as e:
                conn.rollback()
                raise e
            finally:
                cur.close()
                self.pool.putconn(conn)
            
            logger.info(f"Successfully stored {len(table_ids)} tables")
            return table_ids
            
        except Exception as e:
            logger.error(f"Failed to store tables: {e}")
            raise
    
    def store_table(self, file_id: str, file_name: str, table_data: Dict,
                   table_html: str, embedding: List[float], context_before: str,
                   context_after: str, page_number: int, table_index: int,
                   level: str = None, paper: str = None, module: str = None,
                   chapter: str = None, unit: str = None) -> int:
        """Store a single table (legacy method for compatibility)"""
        tables_data = [{
            'file_id': file_id,
            'file_name': file_name,
            'table_data': table_data,
            'table_html': table_html,
            'embedding': embedding,
            'context_before': context_before,
            'context_after': context_after,
            'page_number': page_number,
            'table_index': table_index,
            'level': level,
            'paper': paper,
            'module': module,
            'chapter': chapter,
            'unit': unit
        }]
        
        ids = self.store_tables_batch(tables_data)
        return ids[0] if ids else None
    
    def store_file_metadata(self, file_id: str, file_name: str, appwrite_file_id: str,
                           level: str, paper: str, module: str = None, chapter: str = None,
                           unit: str = None, total_pages: int = 0, source_file: str = None) -> int:
        """Store file metadata with upsert"""
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            
            cur.execute("""
                INSERT INTO file_metadata (file_id, file_name, appwrite_file_id, source_file, level, paper,
                                         module, chapter, unit, total_pages)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (file_id) DO UPDATE SET
                    file_name = EXCLUDED.file_name,
                    appwrite_file_id = EXCLUDED.appwrite_file_id,
                    source_file = EXCLUDED.source_file,
                    level = EXCLUDED.level,
                    paper = EXCLUDED.paper,
                    module = EXCLUDED.module,
                    chapter = EXCLUDED.chapter,
                    unit = EXCLUDED.unit,
                    total_pages = EXCLUDED.total_pages
                RETURNING id;
            """, (file_id, file_name, appwrite_file_id, source_file or '', level, paper, 
                  module or '', chapter or '', unit or '', total_pages))
            
            result = cur.fetchone()
            if result is None:
                raise Exception(f"Failed to insert/update file_metadata for file_id: {file_id}")
            
            metadata_id = result['id']
            conn.commit()
            cur.close()
            self.pool.putconn(conn)
            
            return metadata_id
            
        except Exception as e:
            logger.error(f"Failed to store file metadata: {e}")
            raise
    
    def similarity_search_documents_optimized(self, query_embedding: List[float], top_k: int = 5,
                                            level: str = None, paper: str = None, module: str = None,
                                            chapter: str = None, unit: str = None) -> List[Dict]:
        """Optimized similarity search for documents with better query performance"""
        try:
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            # Build WHERE clause with indexed columns first
            where_conditions = []
            filter_params = []
            
            # Use indexed columns in order of selectivity
            if level:
                where_conditions.append("level = %s")
                filter_params.append(level)
            if paper:
                where_conditions.append("paper = %s")
                filter_params.append(paper)
            if module:
                where_conditions.append("module = %s")
                filter_params.append(module)
            if chapter:
                where_conditions.append("chapter = %s")
                filter_params.append(chapter)
            if unit:
                where_conditions.append("unit = %s")
                filter_params.append(unit)
            
            where_clause = " AND " + " AND ".join(where_conditions) if where_conditions else ""
            
            # Optimized query with better index usage
            query = f"""
                SELECT id, file_id, file_name, content, metadata, chunk_index,
                       level, paper, module, chapter, unit,
                       1 - (embedding <=> %s) as similarity
                FROM documents
                WHERE embedding IS NOT NULL {where_clause}
                ORDER BY embedding <=> %s
                LIMIT %s;
            """
            
            # Convert embedding to string format for pgvector
            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            # Build params in correct order
            params = [embedding_str] + filter_params + [embedding_str, top_k]
            
            cur.execute(query, params)
            results = cur.fetchall() or []
            
            cur.close()
            self.pool.putconn(conn)
            
            return [dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"Document similarity search failed: {e}")
            return []
    
    def similarity_search_tables_optimized(self, query_embedding: List[float], top_k: int = 5,
                                         level: str = None, paper: str = None, module: str = None,
                                         chapter: str = None, unit: str = None) -> List[Dict]:
        """Optimized similarity search for tables with better query performance"""
        try:
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            # Build WHERE clause with indexed columns first
            where_conditions = []
            filter_params = []
            
            if level:
                where_conditions.append("level = %s")
                filter_params.append(level)
            if paper:
                where_conditions.append("paper = %s")
                filter_params.append(paper)
            if module:
                where_conditions.append("module = %s")
                filter_params.append(module)
            if chapter:
                where_conditions.append("chapter = %s")
                filter_params.append(chapter)
            if unit:
                where_conditions.append("unit = %s")
                filter_params.append(unit)
            
            where_clause = " AND " + " AND ".join(where_conditions) if where_conditions else ""
            
            # Optimized query
            query = f"""
                SELECT id, file_id, file_name, table_data, table_html, context_before,
                       context_after, page_number, table_index, level, paper, module,
                       chapter, unit, 1 - (embedding <=> %s) as similarity
                FROM tables
                WHERE embedding IS NOT NULL {where_clause}
                ORDER BY embedding <=> %s
                LIMIT %s;
            """
            
            # Convert embedding to string format for pgvector
            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            # Build params in correct order
            params = [embedding_str] + filter_params + [embedding_str, top_k]
            
            cur.execute(query, params)
            results = cur.fetchall() or []
            
            cur.close()
            self.pool.putconn(conn)
            
            return [dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"Table similarity search failed: {e}")
            return []
    
    # Legacy methods for backward compatibility
    def similarity_search_documents(self, query_embedding: List[float], top_k: int = 5,
                                  level: str = None, paper: str = None, module: str = None,
                                  chapter: str = None, unit: str = None) -> List[Dict]:
        return self.similarity_search_documents_optimized(
            query_embedding, top_k, level, paper, module, chapter, unit
        )
    
    def similarity_search_tables(self, query_embedding: List[float], top_k: int = 5,
                               level: str = None, paper: str = None, module: str = None,
                               chapter: str = None, unit: str = None) -> List[Dict]:
        return self.similarity_search_tables_optimized(
            query_embedding, top_k, level, paper, module, chapter, unit
        )
    
    def get_file_metadata(self, file_id: str = None) -> List[Dict]:
        """Get file metadata with optimized query"""
        try:
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            if file_id:
                cur.execute("SELECT * FROM file_metadata WHERE file_id = %s;", (file_id,))
            else:
                cur.execute("SELECT * FROM file_metadata ORDER BY upload_date DESC;")
            
            results = cur.fetchall()
            cur.close()
            self.pool.putconn(conn)
            
            return [dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"Failed to get file metadata: {e}")
            raise
    
    def update_processing_status(self, file_id: str, status: str):
        """Update file processing status"""
        try:
            conn = self.get_connection()
            cur = conn.cursor()

            cur.execute("""
                UPDATE file_metadata
                SET processing_status = %s, processed_date = CURRENT_TIMESTAMP
                WHERE file_id = %s;
            """, (status, file_id))

            conn.commit()
            cur.close()
            self.pool.putconn(conn)

            logger.info(f"Successfully updated processing status for file_id: {file_id}")

        except Exception as e:
            logger.error(f"Failed to update processing status: {e}")
            raise
    
    async def store_document_chunks_async(self, chunks_data: List[Dict[str, Any]]) -> List[int]:
        """Async wrapper for batch document chunk storage"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            self.store_document_chunks_batch, 
            chunks_data
        )
    
    async def store_tables_async(self, tables_data: List[Dict[str, Any]]) -> List[int]:
        """Async wrapper for batch table storage"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            self.store_tables_batch, 
            tables_data
        )
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics for monitoring"""
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            
            stats = {}
            
            # Document count
            cur.execute("SELECT COUNT(*) as count FROM documents;")
            stats['document_count'] = cur.fetchone()[0]
            
            # Table count
            cur.execute("SELECT COUNT(*) as count FROM tables;")
            stats['table_count'] = cur.fetchone()[0]
            
            # File count
            cur.execute("SELECT COUNT(*) as count FROM file_metadata;")
            stats['file_count'] = cur.fetchone()[0]
            
            # Processing status breakdown
            cur.execute("""
                SELECT processing_status, COUNT(*) as count 
                FROM file_metadata 
                GROUP BY processing_status;
            """)
            stats['processing_status'] = dict(cur.fetchall())
            
            cur.close()
            self.pool.putconn(conn)
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}
    
    def close(self):
        """Close database connections and cleanup"""
        try:
            if self.pool:
                self.pool.closeall()
            if self.executor:
                self.executor.shutdown(wait=True)
            logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")

# Alias for backward compatibility
VectorDatabase = OptimizedVectorDatabase

