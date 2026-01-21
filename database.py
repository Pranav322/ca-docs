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
    def __init__(self, max_connections: int = 50, batch_size: int = 1000):
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
                    1,
                    self.max_connections,
                    dsn=self.connection_url,
                    cursor_factory=RealDictCursor,
                )
            conn = self.pool.getconn()
            return conn
        except Exception as e:
            import traceback

            logger.error(f"Database connection failed: {type(e).__name__}: {e}")
            logger.error(traceback.format_exc())
            raise

    def init_database(self):
        """Initialize database with required tables and extensions"""
        try:
            conn = self.get_connection()
            cur = conn.cursor()

            # Enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # Create curriculum_nodes table (replaces Neo4j graph)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS curriculum_nodes (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL CHECK (type IN ('level', 'paper', 'module', 'chapter', 'unit')),
                    name TEXT NOT NULL,
                    parent_id TEXT REFERENCES curriculum_nodes(id) ON DELETE SET NULL,
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

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
                    source_type TEXT DEFAULT 'ICAI_Module',
                    applicable_attempts TEXT[] DEFAULT ARRAY['May_2026']::TEXT[],
                    node_id TEXT,
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

            # Add new columns if they don't exist (for existing databases)
            cur.execute("""
                ALTER TABLE file_metadata 
                ADD COLUMN IF NOT EXISTS source_file TEXT;
            """)

            # Add new columns to documents table if they don't exist
            cur.execute("""
                ALTER TABLE documents 
                ADD COLUMN IF NOT EXISTS source_type TEXT DEFAULT 'ICAI_Module';
            """)
            cur.execute("""
                ALTER TABLE documents 
                ADD COLUMN IF NOT EXISTS applicable_attempts TEXT[] DEFAULT ARRAY['May_2026']::TEXT[];
            """)
            cur.execute("""
                ALTER TABLE documents 
                ADD COLUMN IF NOT EXISTS node_id TEXT;
            """)

            # Add question context columns to documents table
            cur.execute("""
                ALTER TABLE documents 
                ADD COLUMN IF NOT EXISTS question_id TEXT;
            """)
            cur.execute("""
                ALTER TABLE documents 
                ADD COLUMN IF NOT EXISTS subquestion TEXT;
            """)
            cur.execute("""
                ALTER TABLE documents 
                ADD COLUMN IF NOT EXISTS question_context TEXT;
            """)
            cur.execute("""
                ALTER TABLE documents 
                ADD COLUMN IF NOT EXISTS section_type TEXT DEFAULT 'content';
            """)

            # Add question context columns to tables table
            cur.execute("""
                ALTER TABLE tables 
                ADD COLUMN IF NOT EXISTS question_id TEXT;
            """)
            cur.execute("""
                ALTER TABLE tables 
                ADD COLUMN IF NOT EXISTS subquestion TEXT;
            """)
            cur.execute("""
                ALTER TABLE tables 
                ADD COLUMN IF NOT EXISTS question_context TEXT;
            """)
            cur.execute("""
                ALTER TABLE tables 
                ADD COLUMN IF NOT EXISTS section_type TEXT DEFAULT 'content';
            """)

            # Add MCQ columns to documents table
            cur.execute("""
                ALTER TABLE documents 
                ADD COLUMN IF NOT EXISTS is_mcq BOOLEAN DEFAULT FALSE;
            """)
            cur.execute("""
                ALTER TABLE documents 
                ADD COLUMN IF NOT EXISTS mcq_options JSONB;
            """)
            cur.execute("""
                ALTER TABLE documents 
                ADD COLUMN IF NOT EXISTS correct_answer TEXT;
            """)

            # Create User Progress Table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_progress (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    level TEXT,
                    paper TEXT,
                    status TEXT DEFAULT 'not_started',  -- not_started, in_progress, completed
                    completion_pct FLOAT DEFAULT 0,
                    quiz_score FLOAT,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, node_id)
                );
            """)

            # Add Enrichment Columns to documents (if they don't exist)
            cur.execute("""
                ALTER TABLE documents 
                ADD COLUMN IF NOT EXISTS difficulty TEXT;
            """)
            cur.execute("""
                ALTER TABLE documents 
                ADD COLUMN IF NOT EXISTS estimated_time INTEGER;
            """)
            cur.execute("""
                ALTER TABLE documents 
                ADD COLUMN IF NOT EXISTS topics TEXT[];
            """)
            cur.execute("""
                ALTER TABLE documents 
                ADD COLUMN IF NOT EXISTS question_type TEXT;
            """)

            # Add Q&A Separation and ABC Analysis columns
            cur.execute("""
                ALTER TABLE documents 
                ADD COLUMN IF NOT EXISTS question_text TEXT;
            """)
            cur.execute("""
                ALTER TABLE documents 
                ADD COLUMN IF NOT EXISTS answer_text TEXT;
            """)
            cur.execute("""
                ALTER TABLE documents 
                ADD COLUMN IF NOT EXISTS importance TEXT;  -- 'A', 'B', 'C'
            """)
            cur.execute("""
                ALTER TABLE documents 
                ADD COLUMN IF NOT EXISTS references TEXT[];  -- Array of 'Ind AS 116', 'Sec 185', etc.
            """)

            # Create Study Plan Table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS study_plan (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    target_exam TEXT,
                    daily_hours FLOAT,
                    start_date DATE,
                    end_date DATE,
                    target_level TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Create question logs table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS question_logs (
                    id SERIAL PRIMARY KEY,
                    question TEXT NOT NULL,
                    ip_address TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
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
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_embedding ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_tables_embedding ON tables USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);"
            )

            # Metadata indexes
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_metadata ON documents USING gin(metadata);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_tables_metadata ON tables USING gin(table_data);"
            )

            # Hierarchical indexes
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_hierarchy ON documents(level, paper, module, chapter, unit);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_tables_hierarchy ON tables(level, paper, module, chapter, unit);"
            )

            # File-based indexes
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_file_id ON documents(file_id);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_tables_file_id ON tables(file_id);"
            )

            # Status and date indexes
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_file_metadata_status ON file_metadata(processing_status);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_file_metadata_upload_date ON file_metadata(upload_date);"
            )

            # NEW: Content classification indexes
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_content_type ON documents(content_type);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_source_type ON documents(source_type);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_node_id ON documents(node_id);"
            )

            # NEW: Curriculum nodes indexes
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_curriculum_nodes_parent ON curriculum_nodes(parent_id);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_curriculum_nodes_type ON curriculum_nodes(type);"
            )

            # NEW: GIN index for applicable_attempts array
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_applicable_attempts ON documents USING gin(applicable_attempts);"
            )

            logger.info("Optimized indexes created successfully")

        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")
            raise

    def store_document_chunks_batch(
        self, chunks_data: List[Dict[str, Any]]
    ) -> List[int]:
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
                    cur.execute(
                        """
                        INSERT INTO documents (file_id, file_name, content, embedding, metadata, 
                                             chunk_index, level, paper, module, chapter, unit, 
                                             content_type, source_type, applicable_attempts, node_id,
                                             question_id, subquestion, question_context, section_type,
                                             is_mcq, mcq_options,
                                             difficulty, estimated_time, topics, question_type,
                                             question_text, answer_text, importance, references)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                        (
                            chunk["file_id"],
                            chunk["file_name"],
                            chunk["content"],
                            chunk["embedding"],
                            json.dumps(chunk["metadata"]),
                            chunk["chunk_index"],
                            chunk.get("level", ""),
                            chunk.get("paper", ""),
                            chunk.get("module", ""),
                            chunk.get("chapter", ""),
                            chunk.get("unit", ""),
                            chunk.get("content_type", "text"),
                            chunk.get("source_type", "ICAI_Module"),
                            chunk.get("applicable_attempts", ["May_2026"]),
                            chunk.get("node_id"),
                            chunk.get("question_id", ""),
                            chunk.get("subquestion", ""),
                            chunk.get("question_context", ""),
                            chunk.get("section_type", "content"),
                            chunk.get("is_mcq", False),
                            json.dumps(chunk.get("mcq_options", {})),
                            # Enrichment fields
                            chunk.get("difficulty"),
                            chunk.get("estimated_time"),
                            chunk.get("topics", []),
                            chunk.get("question_type"),
                            # Quiz/ABC fields
                            chunk.get("question_text"),
                            chunk.get("answer_text"),
                            chunk.get("importance"),
                            chunk.get("references", []),
                        ),
                    )

                    result = cur.fetchone()
                    if result:
                        doc_ids.append(
                            result["id"]
                        )  # RealDictCursor returns dict, not tuple

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
            logger.error(
                f"Chunks data length: {len(chunks_data) if chunks_data else 0}"
            )
            if chunks_data:
                logger.error(f"First chunk keys: {list(chunks_data[0].keys())}")
            raise

    def store_document_chunk(
        self,
        file_id: str,
        file_name: str,
        content: str,
        embedding: List[float],
        metadata: Dict,
        chunk_index: int,
        level: str = None,
        paper: str = None,
        module: str = None,
        chapter: str = None,
        unit: str = None,
    ) -> int:
        """Store a single document chunk (legacy method for compatibility)"""
        chunks_data = [
            {
                "file_id": file_id,
                "file_name": file_name,
                "content": content,
                "embedding": embedding,
                "metadata": metadata,
                "chunk_index": chunk_index,
                "level": level,
                "paper": paper,
                "module": module,
                "chapter": chapter,
                "unit": unit,
            }
        ]

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
                    cur.execute(
                        """
                        INSERT INTO tables (file_id, file_name, table_data, table_html, embedding,
                                          context_before, context_after, page_number, table_index,
                                          level, paper, module, chapter, unit,
                                          question_id, subquestion, question_context, section_type)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id;
                    """,
                        (
                            table["file_id"],
                            table["file_name"],
                            json.dumps(table["table_data"]),
                            table.get("table_html", ""),
                            table["embedding"],
                            table.get("context_before", ""),
                            table.get("context_after", ""),
                            table.get("page_number", 0),
                            table.get("table_index", 0),
                            table.get("level", ""),
                            table.get("paper", ""),
                            table.get("module", ""),
                            table.get("chapter", ""),
                            table.get("unit", ""),
                            table.get("question_id", ""),
                            table.get("subquestion", ""),
                            table.get("question_context", ""),
                            table.get("section_type", "content"),
                        ),
                    )

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

    def store_table(
        self,
        file_id: str,
        file_name: str,
        table_data: Dict,
        table_html: str,
        embedding: List[float],
        context_before: str,
        context_after: str,
        page_number: int,
        table_index: int,
        level: str = None,
        paper: str = None,
        module: str = None,
        chapter: str = None,
        unit: str = None,
    ) -> int:
        """Store a single table (legacy method for compatibility)"""
        tables_data = [
            {
                "file_id": file_id,
                "file_name": file_name,
                "table_data": table_data,
                "table_html": table_html,
                "embedding": embedding,
                "context_before": context_before,
                "context_after": context_after,
                "page_number": page_number,
                "table_index": table_index,
                "level": level,
                "paper": paper,
                "module": module,
                "chapter": chapter,
                "unit": unit,
            }
        ]

        ids = self.store_tables_batch(tables_data)
        return ids[0] if ids else None

    def store_file_metadata(
        self,
        file_id: str,
        file_name: str,
        file_path: str,
        level: str,
        paper: str,
        module: str = None,
        chapter: str = None,
        unit: str = None,
        total_pages: int = 0,
    ) -> int:
        """Store file metadata with upsert"""
        try:
            conn = self.get_connection()
            cur = conn.cursor()

            cur.execute(
                """
                INSERT INTO file_metadata (file_id, file_name, source_file, level, paper,
                                         module, chapter, unit, total_pages)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (file_id) DO UPDATE SET
                    file_name = EXCLUDED.file_name,
                    source_file = EXCLUDED.source_file,
                    level = EXCLUDED.level,
                    paper = EXCLUDED.paper,
                    module = EXCLUDED.module,
                    chapter = EXCLUDED.chapter,
                    unit = EXCLUDED.unit,
                    total_pages = EXCLUDED.total_pages
                RETURNING id;
            """,
                (
                    file_id,
                    file_name,
                    file_path,
                    level,
                    paper,
                    module or "",
                    chapter or "",
                    unit or "",
                    total_pages,
                ),
            )

            result = cur.fetchone()
            if result is None:
                raise Exception(
                    f"Failed to insert/update file_metadata for file_id: {file_id}"
                )

            metadata_id = result["id"]
            conn.commit()
            cur.close()
            self.pool.putconn(conn)

            return metadata_id

        except Exception as e:
            logger.error(f"Failed to store file metadata: {e}")
            raise

    def similarity_search_documents_optimized(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        level: str = None,
        paper: str = None,
        module: str = None,
        chapter: str = None,
        unit: str = None,
    ) -> List[Dict]:
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

            where_clause = (
                " AND " + " AND ".join(where_conditions) if where_conditions else ""
            )

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
            embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

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

    def similarity_search_tables_optimized(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        level: str = None,
        paper: str = None,
        module: str = None,
        chapter: str = None,
        unit: str = None,
    ) -> List[Dict]:
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

            where_clause = (
                " AND " + " AND ".join(where_conditions) if where_conditions else ""
            )

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
            embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

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
    def similarity_search_documents(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        level: str = None,
        paper: str = None,
        module: str = None,
        chapter: str = None,
        unit: str = None,
    ) -> List[Dict]:
        return self.similarity_search_documents_optimized(
            query_embedding, top_k, level, paper, module, chapter, unit
        )

    def similarity_search_tables(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        level: str = None,
        paper: str = None,
        module: str = None,
        chapter: str = None,
        unit: str = None,
    ) -> List[Dict]:
        return self.similarity_search_tables_optimized(
            query_embedding, top_k, level, paper, module, chapter, unit
        )

    def get_file_metadata(self, file_id: str = None) -> List[Dict]:
        """Get file metadata with optimized query"""
        try:
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)

            if file_id:
                cur.execute(
                    "SELECT * FROM file_metadata WHERE file_id = %s;", (file_id,)
                )
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

            cur.execute(
                """
                UPDATE file_metadata
                SET processing_status = %s, processed_date = CURRENT_TIMESTAMP
                WHERE file_id = %s;
            """,
                (status, file_id),
            )

            conn.commit()
            cur.close()
            self.pool.putconn(conn)

            logger.info(
                f"Successfully updated processing status for file_id: {file_id}"
            )

        except Exception as e:
            logger.error(f"Failed to update processing status: {e}")
            raise

    async def store_document_chunks_async(
        self, chunks_data: List[Dict[str, Any]]
    ) -> List[int]:
        """Async wrapper for batch document chunk storage"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, self.store_document_chunks_batch, chunks_data
        )

    async def store_tables_async(self, tables_data: List[Dict[str, Any]]) -> List[int]:
        """Async wrapper for batch table storage"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, self.store_tables_batch, tables_data
        )

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics for monitoring"""
        try:
            conn = self.get_connection()
            cur = conn.cursor()

            stats = {}

            # Document count
            cur.execute("SELECT COUNT(*) as count FROM documents;")
            stats["document_count"] = cur.fetchone()[0]

            # Table count
            cur.execute("SELECT COUNT(*) as count FROM tables;")
            stats["table_count"] = cur.fetchone()[0]

            # File count
            cur.execute("SELECT COUNT(*) as count FROM file_metadata;")
            stats["file_count"] = cur.fetchone()[0]

            # Processing status breakdown
            cur.execute("""
                SELECT processing_status, COUNT(*) as count 
                FROM file_metadata 
                GROUP BY processing_status;
            """)
            stats["processing_status"] = dict(cur.fetchall())

            cur.close()
            self.pool.putconn(conn)

            return stats

        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}

    def log_question(self, question: str, ip_address: str):
        """Log user question and IP address"""
        try:
            conn = self.get_connection()
            cur = conn.cursor()

            cur.execute(
                """
                INSERT INTO question_logs (question, ip_address)
                VALUES (%s, %s);
            """,
                (question, ip_address),
            )

            conn.commit()
            cur.close()
            self.pool.putconn(conn)

            logger.info(f"Logged question from {ip_address}")

        except Exception as e:
            logger.error(f"Failed to log question: {e}")
            # Don't raise, we don't want to fail the request just because logging failed

    async def log_question_async(self, question: str, ip_address: str):
        """Async wrapper for logging question"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, lambda: self.log_question(question, ip_address)
        )

    # ============ Async Wrappers for Better Performance ============

    async def similarity_search_documents_async(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        level: str = None,
        paper: str = None,
        module: str = None,
        chapter: str = None,
        unit: str = None,
    ) -> List[Dict]:
        """Async wrapper for similarity search - runs in thread pool to not block event loop"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            lambda: self.similarity_search_documents_optimized(
                query_embedding, top_k, level, paper, module, chapter, unit
            ),
        )

    async def similarity_search_tables_async(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        level: str = None,
        paper: str = None,
        module: str = None,
        chapter: str = None,
        unit: str = None,
    ) -> List[Dict]:
        """Async wrapper for table similarity search - runs in thread pool to not block event loop"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            lambda: self.similarity_search_tables_optimized(
                query_embedding, top_k, level, paper, module, chapter, unit
            ),
        )

    async def get_file_metadata_async(self, file_id: str = None) -> List[Dict]:
        """Async wrapper for getting file metadata"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, lambda: self.get_file_metadata(file_id)
        )

    async def get_database_stats_async(self) -> Dict[str, Any]:
        """Async wrapper for getting database stats"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.get_database_stats)

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
