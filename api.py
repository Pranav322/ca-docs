"""
FastAPI Backend for CA RAG Assistant
Replaces Streamlit UI with REST API endpoints
"""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any
import logging
import tempfile
import time
import os
from datetime import datetime

# Import schemas
from schemas import (
    QuestionRequest, QuestionResponse, AnswerSources, AnswerMetadata,
    SourceDocument, SourceTable,
    DocumentUploadResponse, DocumentUploadMetadata,
    FileMetadata, FileListResponse, FileStatsResponse,
    CurriculumResponse, HealthResponse, ErrorResponse
)

# Import core modules
from database import OptimizedVectorDatabase
from pdf_processor import OptimizedPDFProcessor
from embeddings import OptimizedEmbeddingManager
from table_processor import TableProcessor
from rag_pipeline import RAGPipeline
from appwrite_client import AppwriteClient
from utils import FileUtils, ValidationUtils
from curriculum_manager import curriculum_manager
from cache_manager import cache_manager, search_cache, file_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============ Application State ============

class AppState:
    """Global application state - replaces st.session_state"""
    def __init__(self):
        self.vector_db: Optional[OptimizedVectorDatabase] = None
        self.pdf_processor: Optional[OptimizedPDFProcessor] = None
        self.table_processor: Optional[TableProcessor] = None
        self.embedding_manager: Optional[OptimizedEmbeddingManager] = None
        self.rag_pipeline: Optional[RAGPipeline] = None
        self.appwrite_client: Optional[AppwriteClient] = None
        self.initialized: bool = False
    
    def initialize(self):
        """Initialize all components"""
        if self.initialized:
            return
        
        logger.info("Initializing CA RAG Assistant components...")
        self.vector_db = OptimizedVectorDatabase(max_connections=20, batch_size=1000)
        self.pdf_processor = OptimizedPDFProcessor(max_workers=6)
        self.table_processor = TableProcessor()
        self.embedding_manager = OptimizedEmbeddingManager(max_concurrent_requests=15)
        self.rag_pipeline = RAGPipeline()
        self.appwrite_client = AppwriteClient()
        self.initialized = True
        logger.info("CA RAG Assistant initialized successfully!")


# Global app state
app_state = AppState()


def get_state() -> AppState:
    """Dependency to get app state"""
    if not app_state.initialized:
        app_state.initialize()
    return app_state


# ============ Lifespan Management ============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - initialize on startup, cleanup on shutdown"""
    # Startup
    logger.info("Starting CA RAG Assistant API...")
    app_state.initialize()
    yield
    # Shutdown
    logger.info("Shutting down CA RAG Assistant API...")
    if app_state.vector_db:
        app_state.vector_db.close()


# ============ FastAPI App ============

app = FastAPI(
    title="CA RAG Assistant API",
    description="REST API for CA Study Material Question Answering with RAG",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080",
        "https://cask.vercel.app",
        "https://cask-git-main-pranavs-projects.vercel.app",
        "https://cask.pranavbuilds.tech"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ Health Check ============

@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0"
    )


# ============ Question Answering ============

@app.post("/api/questions/ask", response_model=QuestionResponse, tags=["Questions"])
async def ask_question(
    request: QuestionRequest, 
    background_tasks: BackgroundTasks,
    raw_request: Request,
    state: AppState = Depends(get_state)
):
    """
    Ask a question about CA study materials.
    Uses RAG pipeline to find relevant documents and generate an answer.
    """
    start_time = time.time()
    
    # Log question + IP
    try:
        # Get real IP (handles Cloudflare/Nginx proxies)
        client_ip = raw_request.headers.get("cf-connecting-ip")
        if not client_ip:
            forwarded = raw_request.headers.get("x-forwarded-for")
            if forwarded:
                client_ip = forwarded.split(",")[0].strip()
            else:
                client_ip = raw_request.client.host if raw_request.client else "unknown"
                
        background_tasks.add_task(state.vector_db.log_question, request.question, client_ip)
    except Exception as e:
        logger.error(f"Failed to schedule logging task: {e}")

    try:
        # Check cache first
        filters = {
            'level': request.level,
            'paper': request.paper,
            'module': request.module,
            'chapter': request.chapter,
            'unit': request.unit
        }
        
        search_type = 'documents' if request.include_tables else 'documents_only'
        cached_result = search_cache.get_search_result(request.question, filters, search_type)
        
        if cached_result:
            logger.info(f"Cache hit for question: {request.question[:50]}...")
            answer_data = cached_result
        else:
            # Get answer from RAG pipeline
            answer_data = state.rag_pipeline.answer_question(
                question=request.question,
                level=request.level,
                paper=request.paper,
                module=request.module,
                chapter=request.chapter,
                unit=request.unit,
                include_tables=request.include_tables
            )
            
            # Cache the results
            search_cache.set_search_result(request.question, filters, answer_data, search_type)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Format response
        sources = answer_data.get('sources', {})
        doc_sources = [
            SourceDocument(
                file_name=doc['file_name'],
                level=doc['level'],
                paper=doc['paper'],
                chapter=doc.get('chapter'),
                similarity=doc['similarity'],
                snippet=doc['snippet']
            )
            for doc in sources.get('documents', [])
        ]
        
        table_sources = [
            SourceTable(
                file_name=table['file_name'],
                page_number=table['page_number'],
                level=table['level'],
                paper=table['paper'],
                rows=table['rows'],
                cols=table['cols'],
                similarity=table['similarity'],
                context=table.get('context')
            )
            for table in sources.get('tables', [])
        ]
        
        return QuestionResponse(
            answer=answer_data['answer'],
            confidence=answer_data.get('confidence', 0),
            sources=AnswerSources(documents=doc_sources, tables=table_sources),
            metadata=AnswerMetadata(
                documents_found=answer_data['metadata'].get('documents_found', 0),
                tables_found=answer_data['metadata'].get('tables_found', 0),
                processing_time_ms=processing_time
            ),
            suggestions=answer_data.get('suggestions', [])
        )
        
    except Exception as e:
        logger.error(f"Question answering error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


from fastapi.responses import StreamingResponse

@app.post("/api/questions/ask/stream", tags=["Questions"])
async def ask_question_stream(
    request: QuestionRequest, 
    background_tasks: BackgroundTasks,
    raw_request: Request,
    state: AppState = Depends(get_state)
):
    """
    Ask a question with streaming response.
    Returns Server-Sent Events (SSE) with tokens as they are generated.
    
    Event types:
    - metadata: Contains sources, confidence, and document counts
    - token: Contains a text chunk of the answer
    - done: Signals completion
    - error: Contains error message if something went wrong
    """
    # Log question + IP
    try:
        # Get real IP (handles Cloudflare/Nginx proxies)
        client_ip = raw_request.headers.get("cf-connecting-ip")
        if not client_ip:
            forwarded = raw_request.headers.get("x-forwarded-for")
            if forwarded:
                client_ip = forwarded.split(",")[0].strip()
            else:
                client_ip = raw_request.client.host if raw_request.client else "unknown"
                
        background_tasks.add_task(state.vector_db.log_question, request.question, client_ip)
    except Exception as e:
        logger.error(f"Failed to schedule logging task: {e}")

    def generate():
        try:
            for chunk in state.rag_pipeline.answer_question_stream(
                question=request.question,
                level=request.level,
                paper=request.paper,
                module=request.module,
                chapter=request.chapter,
                unit=request.unit,
                include_tables=request.include_tables
            ):
                yield chunk
        except Exception as e:
            import json
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


# ============ Document Upload ============

@app.post("/api/documents/upload", response_model=DocumentUploadResponse, tags=["Documents"])
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    level: str = Form(...),
    paper: str = Form(...),
    module: Optional[str] = Form(None),
    chapter: Optional[str] = Form(None),
    unit: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    state: AppState = Depends(get_state)
):
    """
    Upload a PDF document with curriculum metadata.
    Processing happens in the background.
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Generate file ID
        file_id = FileUtils.generate_file_id(tmp_file_path)
        sanitized_filename = FileUtils.sanitize_filename(file.filename)
        
        # Prepare metadata
        metadata = {
            'level': level,
            'paper': paper,
            'module': module,
            'chapter': chapter,
            'unit': unit,
            'file_name': sanitized_filename,
            'description': description,
            'tags': [tag.strip() for tag in tags.split(',') if tag.strip()] if tags else []
        }
        
        # Validate metadata
        validated_metadata = ValidationUtils.validate_metadata(metadata)
        
        # Process file in background
        background_tasks.add_task(
            process_document_background,
            file_id=file_id,
            tmp_file_path=tmp_file_path,
            filename=sanitized_filename,
            metadata=validated_metadata,
            state=state
        )
        
        return DocumentUploadResponse(
            success=True,
            file_id=file_id,
            file_name=sanitized_filename,
            message="Document uploaded successfully. Processing in background.",
            pages=0,
            chunks=0,
            tables=0
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_document_background(
    file_id: str,
    tmp_file_path: str,
    filename: str,
    metadata: Dict[str, Any],
    state: AppState
):
    """Background task to process uploaded document"""
    try:
        logger.info(f"Starting background processing for {filename}")
        
        # Validate PDF
        if not FileUtils.validate_pdf_file(tmp_file_path):
            logger.error(f"Invalid PDF file: {filename}")
            return
        
        # Upload to Appwrite
        appwrite_file_id = state.appwrite_client.upload_file(tmp_file_path, filename)
        
        # Store initial metadata
        state.vector_db.store_file_metadata(
            file_id=file_id,
            file_name=filename,
            appwrite_file_id=appwrite_file_id,
            level=metadata['level'],
            paper=metadata['paper'],
            module=metadata.get('module'),
            chapter=metadata.get('chapter'),
            unit=metadata.get('unit')
        )
        
        # Extract PDF content
        pdf_results = state.pdf_processor.extract_text_and_tables(tmp_file_path)
        
        # Update page count
        total_pages = pdf_results['metadata']['total_pages']
        state.vector_db.store_file_metadata(
            file_id=file_id,
            file_name=filename,
            appwrite_file_id=appwrite_file_id,
            level=metadata['level'],
            paper=metadata['paper'],
            module=metadata.get('module'),
            chapter=metadata.get('chapter'),
            unit=metadata.get('unit'),
            total_pages=total_pages
        )
        
        # Process text chunks
        if pdf_results['text_chunks']:
            tables_info = [{'extraction_method': t.get('extraction_method', ''), 'rows': t.get('rows', 0)}
                          for t in pdf_results['tables']]
            
            processed_chunks = []
            for chunk_data in pdf_results['text_chunks']:
                chunk_content = chunk_data['content']
                table_aware_chunks = state.table_processor.chunk_table_aware_text(
                    chunk_content, tables_info, chunk_size=1000, overlap=200
                )
                processed_chunks.extend(table_aware_chunks)
            
            # Generate embeddings
            chunk_embeddings = state.embedding_manager.process_document_chunks(
                processed_chunks, metadata
            )
            
            # Batch store chunks
            chunks_batch_data = []
            for i, chunk in enumerate(chunk_embeddings):
                chunks_batch_data.append({
                    'file_id': file_id,
                    'file_name': filename,
                    'content': chunk['content'],
                    'embedding': chunk['embedding'],
                    'metadata': chunk['metadata'],
                    'chunk_index': i,
                    'level': metadata['level'],
                    'paper': metadata['paper'],
                    'module': metadata.get('module'),
                    'chapter': metadata.get('chapter'),
                    'unit': metadata.get('unit')
                })
            
            state.vector_db.store_document_chunks_batch(chunks_batch_data)
        
        # Process tables
        if pdf_results['tables']:
            table_embeddings = state.embedding_manager.process_tables(
                pdf_results['tables'], metadata
            )
            
            tables_batch_data = []
            for i, table in enumerate(table_embeddings):
                tables_batch_data.append({
                    'file_id': file_id,
                    'file_name': filename,
                    'table_data': table['data'] if 'data' in table else {},
                    'table_html': table.get('html', ''),
                    'embedding': table['embedding'],
                    'context_before': table.get('context_before', ''),
                    'context_after': table.get('context_after', ''),
                    'page_number': table.get('page_number', 0),
                    'table_index': i,
                    'level': metadata['level'],
                    'paper': metadata['paper'],
                    'module': metadata.get('module'),
                    'chapter': metadata.get('chapter'),
                    'unit': metadata.get('unit')
                })
            
            state.vector_db.store_tables_batch(tables_batch_data)
        
        # Update processing status
        state.vector_db.update_processing_status(file_id, "completed")
        
        # Cleanup
        FileUtils.cleanup_temp_file(tmp_file_path)
        
        logger.info(f"Successfully processed {filename}: {len(pdf_results['text_chunks'])} chunks, {len(pdf_results['tables'])} tables")
        
    except Exception as e:
        logger.error(f"Background processing error for {filename}: {e}")
        try:
            state.vector_db.update_processing_status(file_id, "failed")
            FileUtils.cleanup_temp_file(tmp_file_path)
        except:
            pass


# ============ File Management ============

@app.get("/api/files", response_model=FileListResponse, tags=["Files"])
async def list_files(
    level: Optional[str] = None,
    paper: Optional[str] = None,
    status: Optional[str] = None,
    search: Optional[str] = None,
    state: AppState = Depends(get_state)
):
    """List all files with optional filters"""
    try:
        all_files = state.vector_db.get_file_metadata()
        
        # Apply filters
        filtered_files = all_files
        
        if level:
            filtered_files = [f for f in filtered_files if f.get('level') == level]
        if paper:
            filtered_files = [f for f in filtered_files if f.get('paper') == paper]
        if status:
            filtered_files = [f for f in filtered_files if f.get('processing_status') == status]
        if search:
            filtered_files = [f for f in filtered_files if search.lower() in f.get('file_name', '').lower()]
        
        return FileListResponse(
            files=[
                FileMetadata(
                    file_id=f['file_id'],
                    file_name=f['file_name'],
                    level=f['level'],
                    paper=f['paper'],
                    module=f.get('module'),
                    chapter=f.get('chapter'),
                    unit=f.get('unit'),
                    total_pages=f.get('total_pages', 0),
                    processing_status=f.get('processing_status', 'unknown'),
                    upload_date=f.get('upload_date')
                )
                for f in filtered_files
            ],
            total=len(filtered_files)
        )
        
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/files/stats", response_model=FileStatsResponse, tags=["Files"])
async def get_file_stats(state: AppState = Depends(get_state)):
    """Get file statistics"""
    try:
        all_files = state.vector_db.get_file_metadata()
        db_stats = state.vector_db.get_database_stats()
        
        return FileStatsResponse(
            total_files=len(all_files),
            completed=len([f for f in all_files if f.get('processing_status') == 'completed']),
            pending=len([f for f in all_files if f.get('processing_status') == 'pending']),
            failed=len([f for f in all_files if f.get('processing_status') == 'failed']),
            total_pages=sum(f.get('total_pages', 0) for f in all_files),
            total_chunks=db_stats.get('document_count', 0),
            total_tables=db_stats.get('table_count', 0)
        )
        
    except Exception as e:
        logger.error(f"Error getting file stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/files/{file_id}", response_model=FileMetadata, tags=["Files"])
async def get_file(file_id: str, state: AppState = Depends(get_state)):
    """Get a specific file's metadata"""
    try:
        files = state.vector_db.get_file_metadata(file_id)
        
        if not files:
            raise HTTPException(status_code=404, detail="File not found")
        
        f = files[0]
        return FileMetadata(
            file_id=f['file_id'],
            file_name=f['file_name'],
            level=f['level'],
            paper=f['paper'],
            module=f.get('module'),
            chapter=f.get('chapter'),
            unit=f.get('unit'),
            total_pages=f.get('total_pages', 0),
            processing_status=f.get('processing_status', 'unknown'),
            upload_date=f.get('upload_date')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting file {file_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ Curriculum ============

@app.get("/api/curriculum", response_model=CurriculumResponse, tags=["Curriculum"])
async def get_curriculum():
    """Get the full CA curriculum hierarchy"""
    try:
        # curriculum_data is the attribute, not get_curriculum_hierarchy()
        hierarchy = curriculum_manager.curriculum_data
        return CurriculumResponse(levels=hierarchy)
    except Exception as e:
        logger.error(f"Error getting curriculum: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ Cache Management ============

@app.get("/api/cache/stats", tags=["Cache"])
async def get_cache_stats():
    """Get cache statistics"""
    return cache_manager.get_stats()


@app.post("/api/cache/clear", tags=["Cache"])
async def clear_cache():
    """Clear all caches"""
    try:
        cache_manager.clear_all()
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))
