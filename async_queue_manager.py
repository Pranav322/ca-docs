import asyncio
import aiofiles
import uuid
import tempfile
import os
import time
import logging
from typing import Dict, List, Any, Optional
from database import VectorDatabase
from pdf_processor import PDFProcessor
from embeddings import EmbeddingManager
from appwrite_client import AppwriteClient
from utils import FileUtils, ValidationUtils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AsyncFileQueueManager:
    """Async file processing queue manager with concurrent processing"""
    
    def __init__(self, max_workers: int = 4):
        self.queue = asyncio.Queue(maxsize=100)  # Max 100 files in queue
        self.active_sessions = {}  # session_id -> session_data
        self.workers = []
        self.max_workers = max_workers
        self.is_running = False
        
        # Initialize components
        self.vector_db = VectorDatabase()
        self.pdf_processor = PDFProcessor()
        self.embedding_manager = EmbeddingManager()
        self.appwrite_client = AppwriteClient()
        
        # Processing stats
        self.stats = {
            'files_processed': 0,
            'files_failed': 0,
            'total_processing_time': 0
        }
    
    async def start_workers(self):
        """Start background worker tasks"""
        if self.is_running:
            return
            
        logger.info(f"Starting {self.max_workers} async workers...")
        self.is_running = True
        
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
        
        logger.info("Async workers started successfully")
    
    async def stop_workers(self):
        """Stop all workers"""
        logger.info("Stopping async workers...")
        self.is_running = False
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()
        
        logger.info("Async workers stopped")
    
    async def enqueue_batch(self, files_data: List[Dict]) -> str:
        """Add multiple files to processing queue"""
        session_id = str(uuid.uuid4())

        try:
            # Track session
            self.active_sessions[session_id] = {
                'total_files': len(files_data),
                'queued_files': 0,
                'start_time': time.time()
            }

            # Add files to queue and database
            for file_data in files_data:
                file_id = str(uuid.uuid4())
                file_data['file_id'] = file_id
                file_data['session_id'] = session_id

                # Add to database queue (this will create the file_metadata entry)
                self.vector_db.add_file_to_queue(
                    session_id=session_id,
                    file_id=file_id,
                    file_name=file_data['file_name'],
                    metadata=file_data['metadata']
                )

                # Add to processing queue
                logger.info(f"Adding file {file_data['file_name']} to processing queue")
                await self.queue.put(file_data)
                self.active_sessions[session_id]['queued_files'] += 1
                logger.info(f"File {file_data['file_name']} added to queue. Queue size: {self.queue.qsize()}")

            logger.info(f"Batch queued: {len(files_data)} files in session {session_id}")
            logger.info(f"Final queue size: {self.queue.qsize()}")
            return session_id

        except Exception as e:
            logger.error(f"Failed to enqueue batch: {e}")
            raise
    
    async def _worker(self, worker_name: str):
        """Background worker for processing files"""
        logger.info(f"{worker_name} started")

        while self.is_running:
            try:
                # Get file from queue with timeout
                try:
                    logger.info(f"{worker_name} waiting for file from queue...")
                    file_data = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                    logger.info(f"{worker_name} got file: {file_data.get('file_name', 'Unknown')}")
                except asyncio.TimeoutError:
                    logger.debug(f"{worker_name} timeout waiting for file")
                    continue  # Check if still running

                # Process the file
                logger.info(f"{worker_name} processing file: {file_data.get('file_name', 'Unknown')}")
                await self._process_file(file_data, worker_name)
                self.queue.task_done()
                logger.info(f"{worker_name} completed processing file: {file_data.get('file_name', 'Unknown')}")

            except asyncio.CancelledError:
                logger.info(f"{worker_name} cancelled")
                break
            except Exception as e:
                logger.error(f"{worker_name} error: {e}")
                await asyncio.sleep(1)  # Brief pause before retry

        logger.info(f"{worker_name} stopped")
    
    async def _process_file(self, file_data: Dict, worker_name: str):
        """Process single file asynchronously"""
        file_id = file_data['file_id']
        session_id = file_data['session_id']
        file_name = file_data['file_name']
        uploaded_file = file_data['uploaded_file']
        metadata = file_data['metadata']

        start_time = time.time()
        temp_file_path = None

        try:
            logger.info(f"{worker_name} processing {file_name}")

            # Update status to processing
            self.vector_db.update_processing_status(file_id, 'processing')

            # Step 1: Create temporary file and validate
            temp_file_path = await self._create_temp_file(uploaded_file)

            if not FileUtils.validate_pdf_file(temp_file_path):
                raise Exception("Invalid PDF file")

            # Step 2: Upload to Appwrite
            appwrite_file_id = await self._async_upload_file(temp_file_path, file_name)

            # Step 3: Extract PDF content
            pdf_results = await self._async_process_pdf(temp_file_path)

            # Step 4: Generate embeddings in batch
            processed_chunks, processed_tables = await self._async_generate_embeddings(
                pdf_results, metadata
            )

            # Step 5: Store in database
            await self._async_store_data(
                file_id, file_name, appwrite_file_id, processed_chunks,
                processed_tables, metadata, pdf_results.get('metadata', {})
            )

            # Complete processing
            self.vector_db.update_processing_status(file_id, 'completed')

            # Update stats
            processing_time = time.time() - start_time
            self.stats['files_processed'] += 1
            self.stats['total_processing_time'] += processing_time

            logger.info(f"{worker_name} completed {file_name} in {processing_time:.1f}s")

        except Exception as e:
            logger.error(f"{worker_name} failed processing {file_name}: {e}")
            self.vector_db.update_processing_status(file_id, 'failed')
            self.stats['files_failed'] += 1

        finally:
            # Cleanup temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                FileUtils.cleanup_temp_file(temp_file_path)
    
    async def _create_temp_file(self, uploaded_file) -> str:
        """Create temporary file from uploaded file"""
        try:
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            temp_file_path = temp_file.name
            
            # Write uploaded file content to temp file asynchronously
            async with aiofiles.open(temp_file_path, 'wb') as f:
                await f.write(uploaded_file.getvalue())
            
            return temp_file_path
            
        except Exception as e:
            logger.error(f"Failed to create temp file: {e}")
            raise
    
    async def _async_upload_file(self, file_path: str, file_name: str) -> str:
        """Upload file to Appwrite asynchronously"""
        try:
            # Run in thread pool to not block event loop
            loop = asyncio.get_event_loop()
            appwrite_file_id = await loop.run_in_executor(
                None,
                self.appwrite_client.upload_file,
                file_path,
                file_name
            )
            return appwrite_file_id
            
        except Exception as e:
            logger.error(f"Failed to upload file: {e}")
            raise
    
    async def _async_process_pdf(self, file_path: str) -> Dict[str, Any]:
        """Process PDF content asynchronously"""
        try:
            # Run in thread pool to not block event loop
            loop = asyncio.get_event_loop()
            pdf_results = await loop.run_in_executor(
                None,
                self.pdf_processor.extract_text_and_tables,
                file_path
            )
            return pdf_results
            
        except Exception as e:
            logger.error(f"Failed to process PDF: {e}")
            raise
    
    async def _async_generate_embeddings(self, pdf_results: Dict, metadata: Dict) -> tuple:
        """Generate embeddings asynchronously with batching"""
        try:
            # Process text chunks
            processed_chunks = []
            if pdf_results.get('text_chunks'):
                # Run in thread pool
                loop = asyncio.get_event_loop()
                processed_chunks = await loop.run_in_executor(
                    None,
                    self.embedding_manager.process_document_chunks,
                    pdf_results['text_chunks'],
                    metadata
                )
            
            # Process tables
            processed_tables = []
            if pdf_results.get('tables'):
                # Run in thread pool
                loop = asyncio.get_event_loop()
                processed_tables = await loop.run_in_executor(
                    None,
                    self.embedding_manager.process_tables,
                    pdf_results['tables'],
                    metadata
                )
            
            return processed_chunks, processed_tables
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    async def _async_store_data(self, file_id: str, file_name: str, appwrite_file_id: str,
                               processed_chunks: List[Dict], processed_tables: List[Dict],
                               metadata: Dict, pdf_metadata: Dict):
        """Store processed data in database asynchronously"""
        try:
            loop = asyncio.get_event_loop()
            
            # Store file metadata
            await loop.run_in_executor(
                None,
                self.vector_db.store_file_metadata,
                file_id, file_name, appwrite_file_id,
                metadata['level'], metadata['paper'],
                metadata.get('module'), metadata.get('chapter'), metadata.get('unit'),
                pdf_metadata.get('total_pages', 0)
            )
            
            # Store document chunks
            for i, chunk in enumerate(processed_chunks):
                await loop.run_in_executor(
                    None,
                    self.vector_db.store_document_chunk,
                    file_id, file_name, chunk['content'], chunk['embedding'],
                    chunk['metadata'], i,
                    metadata['level'], metadata['paper'],
                    metadata.get('module'), metadata.get('chapter'), metadata.get('unit')
                )
            
            # Store tables
            for i, table in enumerate(processed_tables):
                await loop.run_in_executor(
                    None,
                    self.vector_db.store_table,
                    file_id, file_name, table.get('data', {}), table.get('html', ''),
                    table['embedding'], table.get('context_before', ''),
                    table.get('context_after', ''), table.get('page_number', 0), i,
                    metadata['level'], metadata['paper'],
                    metadata.get('module'), metadata.get('chapter'), metadata.get('unit')
                )
            
            # Update processing status in original table
            await loop.run_in_executor(
                None,
                self.vector_db.update_processing_status,
                file_id, 'completed'
            )
            
        except Exception as e:
            logger.error(f"Failed to store data: {e}")
            raise
    
    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get real-time session status"""
        return self.vector_db.get_session_status(session_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        stats = self.stats.copy()
        if stats['files_processed'] > 0:
            stats['avg_processing_time'] = stats['total_processing_time'] / stats['files_processed']
        else:
            stats['avg_processing_time'] = 0
        return stats
    
# Global instance
async_queue_manager = None

def get_queue_manager(max_workers: int = 4) -> AsyncFileQueueManager:
    """Get or create global queue manager instance"""
    global async_queue_manager
    if async_queue_manager is None:
        async_queue_manager = AsyncFileQueueManager(max_workers)
    return async_queue_manager