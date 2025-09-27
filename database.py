import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
import json
from typing import List, Dict, Any, Optional
import logging
from config import DATABASE_URL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDatabase:
    def __init__(self):
        self.connection_url = DATABASE_URL
        self.init_database()
    
    def get_connection(self):
        """Get database connection"""
        try:
            conn = psycopg2.connect(self.connection_url, cursor_factory=RealDictCursor)
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
            
            # Create documents table for text chunks
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
            
            # Create tables table for extracted tables
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
            
            # Add source_file column if it doesn't exist (for existing databases)
            cur.execute("""
                ALTER TABLE file_metadata 
                ADD COLUMN IF NOT EXISTS source_file TEXT;
            """)
            
            # Create indexes for better performance
            cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_embedding ON documents USING ivfflat (embedding vector_cosine_ops);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_tables_embedding ON tables USING ivfflat (embedding vector_cosine_ops);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_metadata ON documents USING gin(metadata);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_level ON documents(level);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_paper ON documents(paper);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_tables_level ON tables(level);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_tables_paper ON tables(paper);")
            
            conn.commit()
            cur.close()
            conn.close()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def store_document_chunk(self, file_id: str, file_name: str, content: str, 
                           embedding: List[float], metadata: Dict, chunk_index: int,
                           level: str = None, paper: str = None, module: str = None,
                           chapter: str = None, unit: str = None) -> int:
        """Store a document chunk with its embedding"""
        try:
            conn = psycopg2.connect(self.connection_url)  # Don't use RealDictCursor here
            cur = conn.cursor()
            
            cur.execute("""
                INSERT INTO documents (file_id, file_name, content, embedding, metadata, 
                                     chunk_index, level, paper, module, chapter, unit)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
            """, (file_id, file_name, content, embedding, json.dumps(metadata), 
                  chunk_index, level or '', paper or '', module or '', chapter or '', unit or ''))
            
            doc_id = cur.fetchone()[0]
            conn.commit()
            cur.close()
            conn.close()
            
            return doc_id
            
        except Exception as e:
            error_msg = f"Failed to store document chunk: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg) from e
    
    def store_table(self, file_id: str, file_name: str, table_data: Dict,
                   table_html: str, embedding: List[float], context_before: str,
                   context_after: str, page_number: int, table_index: int,
                   level: str = None, paper: str = None, module: str = None,
                   chapter: str = None, unit: str = None) -> int:
        """Store extracted table with its embedding"""
        try:
            conn = psycopg2.connect(self.connection_url)  # Don't use RealDictCursor here
            cur = conn.cursor()
            
            cur.execute("""
                INSERT INTO tables (file_id, file_name, table_data, table_html, embedding,
                                  context_before, context_after, page_number, table_index,
                                  level, paper, module, chapter, unit)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
            """, (file_id, file_name, json.dumps(table_data), table_html, embedding,
                  context_before, context_after, page_number, table_index,
                  level or '', paper or '', module or '', chapter or '', unit or ''))
            
            table_id = cur.fetchone()[0]
            conn.commit()
            cur.close()
            conn.close()
            
            return table_id
            
        except Exception as e:
            error_msg = f"Failed to store table: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg) from e
    
    def store_file_metadata(self, file_id: str, file_name: str, appwrite_file_id: str,
                           level: str, paper: str, module: str = None, chapter: str = None,
                           unit: str = None, total_pages: int = 0, source_file: str = None) -> int:
        """Store file metadata"""
        try:
            conn = psycopg2.connect(self.connection_url)  # Don't use RealDictCursor here
            cur = conn.cursor()
            
            logger.info(f"=== STORE_FILE_METADATA CALLED ===")
            logger.info(f"file_id: '{file_id}' (type: {type(file_id)})")
            logger.info(f"file_name: '{file_name}' (type: {type(file_name)})")
            logger.info(f"appwrite_file_id: '{appwrite_file_id}' (type: {type(appwrite_file_id)})")
            logger.info(f"level: '{level}' (type: {type(level)})")
            logger.info(f"paper: '{paper}' (type: {type(paper)})")
            logger.info(f"module: '{module}' (type: {type(module)})")
            logger.info(f"chapter: '{chapter}' (type: {type(chapter)})")
            logger.info(f"unit: '{unit}' (type: {type(unit)})")
            logger.info(f"total_pages: {total_pages} (type: {type(total_pages)})")
            logger.info(f"source_file: '{source_file}' (type: {type(source_file)})")
            
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
            """, (file_id, file_name, appwrite_file_id, source_file or '', level, paper, module or '', chapter or '', unit or '', total_pages))
            
            result = cur.fetchone()
            if result is None:
                raise Exception(f"Failed to insert/update file_metadata - no ID returned for file_id: {file_id}")
            
            metadata_id = result[0]
            logger.info(f"Successfully stored file metadata with ID: {metadata_id}")
            conn.commit()
            cur.close()
            conn.close()
            
            return metadata_id
            
        except Exception as e:
            # Enhanced debugging
            logger.error(f"EXCEPTION CAUGHT: {repr(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Exception message: '{str(e)}'")
            logger.error(f"Exception args: {e.args}")
            
            # Import traceback for full stack trace
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            # Also log more details for psycopg2 errors
            if hasattr(e, 'pgcode'):
                logger.error(f"PostgreSQL error code: {e.pgcode}")
            if hasattr(e, 'pgerror'):
                logger.error(f"PostgreSQL error: {e.pgerror}")
            
            error_msg = f"Failed to store file metadata - Exception type: {type(e).__name__}, Message: '{str(e)}', Args: {e.args}"
            raise Exception(error_msg) from e
    
    def similarity_search_documents(self, query_embedding: List[float], top_k: int = 5,
                                  level: str = None, paper: str = None, module: str = None,
                                  chapter: str = None, unit: str = None) -> List[Dict]:
        """Search for similar document chunks"""
        try:
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            # Build WHERE clause based on filters
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
            
            query = f"""
                SELECT id, file_id, file_name, content, metadata, chunk_index,
                       level, paper, module, chapter, unit,
                       1 - (embedding <=> %s) as similarity
                FROM documents
                WHERE 1=1 {where_clause}
                ORDER BY embedding <=> %s
                LIMIT %s;
            """
            
            # Convert embedding to string format for pgvector
            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            # Build params in correct order: embedding_str, filter_params..., embedding_str, top_k
            params = [embedding_str] + filter_params + [embedding_str, top_k]
            
            cur.execute(query, params)
            results = cur.fetchall() or []
            
            cur.close()
            conn.close()
            
            return [dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"Document similarity search failed: {e}")
            return []
    
    def similarity_search_tables(self, query_embedding: List[float], top_k: int = 5,
                               level: str = None, paper: str = None, module: str = None,
                               chapter: str = None, unit: str = None) -> List[Dict]:
        """Search for similar tables"""
        try:
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            # Build WHERE clause based on filters
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
            
            query = f"""
                SELECT id, file_id, file_name, table_data, table_html, context_before,
                       context_after, page_number, table_index, level, paper, module,
                       chapter, unit, 1 - (embedding <=> %s) as similarity
                FROM tables
                WHERE 1=1 {where_clause}
                ORDER BY embedding <=> %s
                LIMIT %s;
            """
            
            # Convert embedding to string format for pgvector
            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            # Build params in correct order: embedding_str, filter_params..., embedding_str, top_k
            params = [embedding_str] + filter_params + [embedding_str, top_k]
            
            cur.execute(query, params)
            results = cur.fetchall() or []
            
            cur.close()
            conn.close()
            
            return [dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"Table similarity search failed: {e}")
            return []
    
    def get_file_metadata(self, file_id: str = None) -> List[Dict]:
        """Get file metadata"""
        try:
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            if file_id:
                cur.execute("SELECT * FROM file_metadata WHERE file_id = %s;", (file_id,))
            else:
                cur.execute("SELECT * FROM file_metadata ORDER BY upload_date DESC;")
            
            results = cur.fetchall()
            cur.close()
            conn.close()
            
            return [dict(row) for row in results]
            
        except Exception as e:
            error_msg = f"Failed to get file metadata: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg) from e
    
    def update_processing_status(self, file_id: str, status: str):
        """Update file processing status"""
        try:
            conn = psycopg2.connect(self.connection_url)  # Don't use RealDictCursor here
            cur = conn.cursor()
            
            cur.execute("""
                UPDATE file_metadata 
                SET processing_status = %s, processed_date = CURRENT_TIMESTAMP
                WHERE file_id = %s;
            """, (status, file_id))
            
            conn.commit()
            cur.close()
            conn.close()
            
        except Exception as e:
            error_msg = f"Failed to update processing status: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg) from e
