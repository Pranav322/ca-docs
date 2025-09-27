#!/usr/bin/env python3
"""
Automated CA PDF Batch Ingestion Script

This script automatically processes all PDF files in the ca/ folder,
infers curriculum metadata from folder structure, and processes them
through the full RAG pipeline with parallel processing and retries.
"""

import os
import asyncio
import uuid
import tempfile
import shutil
import logging
import time
from typing import Dict, Any, List, Optional
from pathlib import Path
import concurrent.futures
from dataclasses import dataclass
from enum import Enum

# Import existing components
from database import VectorDatabase
from pdf_processor import PDFProcessor
from embeddings import EmbeddingManager
from appwrite_client import AppwriteClient
from utils import FileUtils, ValidationUtils
from config import CHUNK_SIZE, CHUNK_OVERLAP

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_ingest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProcessingStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"

@dataclass
class FileTask:
    file_path: str
    file_name: str
    file_id: str
    metadata: Dict[str, Any]
    attempt: int = 0
    max_attempts: int = 3
    status: ProcessingStatus = ProcessingStatus.PENDING
    error: Optional[str] = None
    processing_time: float = 0.0

class BatchIngestor:
    def __init__(self, ca_folder_path: str = "ca", max_workers: int = 4, retry_attempts: int = 3):
        """
        Initialize batch ingestor
        
        Args:
            ca_folder_path: Path to ca folder containing PDFs
            max_workers: Number of concurrent workers
            retry_attempts: Max retry attempts per file
        """
        self.ca_folder_path = Path(ca_folder_path).resolve()
        self.max_workers = max_workers
        self.retry_attempts = retry_attempts
        
        # Initialize components
        self.vector_db = VectorDatabase()
        self.pdf_processor = PDFProcessor()
        self.embedding_manager = EmbeddingManager()
        self.appwrite_client = AppwriteClient()
        
        # Processing stats
        self.stats = {
            'total_files': 0,
            'completed': 0,
            'failed': 0,
            'retries': 0,
            'start_time': 0,
            'end_time': 0
        }
        
        # Task tracking
        self.tasks: List[FileTask] = []
        self.completed_files = set()
        self.failed_files = set()
        
        logger.info(f"Batch Ingestor initialized with {max_workers} workers, max {retry_attempts} retries")

    def discover_files(self) -> List[FileTask]:
        """
        Discover all PDF files in ca folder and create tasks with inferred metadata
        """
        logger.info(f"Discovering PDF files in {self.ca_folder_path}")
        
        if not self.ca_folder_path.exists():
            raise FileNotFoundError(f"CA folder not found: {self.ca_folder_path}")
        
        tasks = []
        
        # Walk through all subdirectories
        for pdf_file in self.ca_folder_path.rglob("*.pdf"):
            try:
                # Get relative path from ca folder
                rel_path = pdf_file.relative_to(self.ca_folder_path)
                path_parts = rel_path.parts
                
                # Infer metadata from folder structure (same as ca/path.py)
                metadata = self._build_metadata(path_parts, str(rel_path))
                
                # Validate metadata
                if not self._validate_metadata(metadata):
                    logger.warning(f"Skipping {pdf_file} - invalid metadata: {metadata}")
                    continue
                
                # Create task
                file_id = str(uuid.uuid4())
                task = FileTask(
                    file_path=str(pdf_file),
                    file_name=pdf_file.name,
                    file_id=file_id,
                    metadata=metadata,
                    max_attempts=self.retry_attempts
                )
                
                tasks.append(task)
                logger.debug(f"Added task: {pdf_file.name} -> {metadata}")
                
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {e}")
                continue
        
        logger.info(f"Discovered {len(tasks)} PDF files for processing")
        return tasks

    def _build_metadata(self, path_parts: tuple, file_path: str) -> Dict[str, Any]:
        """
        Build metadata from path parts (same logic as ca/path.py)
        """
        def normalize(name):
            return name.strip().replace(" .", ".").replace("..", ".").replace("  ", " ")
        
        meta = {
            "level": None,
            "paper": None, 
            "module": None,
            "chapter": None,
            "unit": None,
            "source_file": file_path,
            "file_name": path_parts[-1] if path_parts else "unknown.pdf"
        }
        
        # Extract hierarchy from path
        if len(path_parts) >= 1:
            meta["level"] = normalize(path_parts[0])
        if len(path_parts) >= 2:
            meta["paper"] = normalize(path_parts[1])
        if len(path_parts) >= 3 and path_parts[2].lower().startswith("module"):
            meta["module"] = normalize(path_parts[2])
        if len(path_parts) >= 4 and path_parts[3].lower().startswith("chapter"):
            meta["chapter"] = normalize(path_parts[3])
        
        # Check filename for unit/chapter info
        filename = path_parts[-1]
        if filename.lower().startswith("unit"):
            meta["unit"] = normalize(filename.replace(".pdf", ""))
        elif filename.lower().startswith("chapter"):
            meta["chapter"] = normalize(filename.replace(".pdf", ""))
        
        return meta

    def _validate_metadata(self, metadata: Dict[str, Any]) -> bool:
        """
        Validate that metadata has minimum required fields
        """
        required_fields = ['level', 'paper']
        
        for field in required_fields:
            if not metadata.get(field):
                return False
        
        # Validate CA level
        valid_levels = ['Foundation', 'Intermediate', 'Final']
        if metadata['level'] not in valid_levels:
            # Try to match case-insensitive
            for level in valid_levels:
                if metadata['level'].lower() == level.lower():
                    metadata['level'] = level
                    break
            else:
                return False
        
        return True

    def check_existing_files(self, tasks: List[FileTask]) -> List[FileTask]:
        """
        Filter out files that are already processed successfully
        """
        logger.info("Checking for already processed files...")
        
        try:
            existing_files = self.vector_db.get_file_metadata()
            existing_paths = {f['source_file'] for f in existing_files if f['processing_status'] == 'completed'}
            
            new_tasks = []
            for task in tasks:
                rel_path = str(Path(task.file_path).relative_to(self.ca_folder_path))
                if rel_path in existing_paths:
                    logger.info(f"Skipping already processed file: {task.file_name}")
                    self.completed_files.add(task.file_id)
                else:
                    new_tasks.append(task)
            
            logger.info(f"Found {len(existing_paths)} already processed, {len(new_tasks)} new files to process")
            return new_tasks
            
        except Exception as e:
            logger.warning(f"Could not check existing files: {e}. Processing all files.")
            return tasks

    async def process_batch(self, tasks: List[FileTask]) -> None:
        """
        Process all tasks with parallel workers and retry logic
        """
        if not tasks:
            logger.info("No files to process")
            return
        
        self.tasks = tasks
        self.stats['total_files'] = len(tasks)
        self.stats['start_time'] = time.time()
        
        logger.info(f"Starting batch processing of {len(tasks)} files with {self.max_workers} workers")
        
        # Create semaphore to limit concurrent processing
        semaphore = asyncio.Semaphore(self.max_workers)
        
        # Process all tasks concurrently with proper cancellation handling
        try:
            # Use gather to process all tasks concurrently
            await asyncio.gather(*[self._process_with_retry(task, semaphore) for task in tasks])
        except KeyboardInterrupt:
            logger.info("Processing interrupted by user")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during batch processing: {e}")
            raise
        
        self.stats['end_time'] = time.time()
        self._print_final_stats()

    async def _process_with_retry(self, task: FileTask, semaphore: asyncio.Semaphore) -> None:
        """
        Process a single task with retry logic
        """
        async with semaphore:  # Limit concurrent processing
            while task.attempt < task.max_attempts and task.status != ProcessingStatus.COMPLETED:
                task.attempt += 1
                
                if task.attempt > 1:
                    task.status = ProcessingStatus.RETRYING
                    self.stats['retries'] += 1
                    wait_time = 2 ** (task.attempt - 1)  # Exponential backoff
                    logger.info(f"Retrying {task.file_name} (attempt {task.attempt}/{task.max_attempts}) after {wait_time}s delay")
                    await asyncio.sleep(wait_time)
                else:
                    task.status = ProcessingStatus.PROCESSING
                    logger.info(f"Processing {task.file_name} (attempt {task.attempt}/{task.max_attempts})")
                
                try:
                    start_time = time.time()
                    await self._process_single_file(task)
                    task.processing_time = time.time() - start_time
                    task.status = ProcessingStatus.COMPLETED
                    self.completed_files.add(task.file_id)
                    self.stats['completed'] += 1
                    
                    logger.info(f"✅ Completed {task.file_name} in {task.processing_time:.1f}s")
                    break
                    
                except Exception as e:
                    task.error = str(e)
                    logger.error(f"❌ Failed {task.file_name} (attempt {task.attempt}): {e}")
                    
                    if task.attempt >= task.max_attempts:
                        task.status = ProcessingStatus.FAILED
                        self.failed_files.add(task.file_id)
                        self.stats['failed'] += 1
                        logger.error(f"🚫 Permanently failed {task.file_name} after {task.max_attempts} attempts")

    async def _process_single_file(self, task: FileTask) -> None:
        """
        Process a single PDF file through the complete pipeline
        """
        temp_file_path = None
        
        try:
            # Step 1: Validate file
            if not FileUtils.validate_pdf_file(task.file_path):
                raise Exception(f"Invalid PDF file: {task.file_path}")
            
            # Step 2: Create temporary copy for processing
            temp_file_path = await self._create_temp_copy(task.file_path)
            
            # Step 3: Upload to Appwrite storage
            appwrite_file_id = await self._upload_to_appwrite(temp_file_path, task.file_name)
            
            # Step 4: Extract PDF content (text and tables)
            pdf_results = await self._extract_pdf_content(temp_file_path)
            
            # Step 5: Generate embeddings for text chunks and tables
            processed_chunks, processed_tables = await self._generate_embeddings(
                pdf_results, task.metadata
            )
            
            # Step 6: Store everything in vector database
            await self._store_in_database(
                task.file_id,
                task.file_name, 
                appwrite_file_id,
                processed_chunks,
                processed_tables,
                task.metadata,
                pdf_results.get('metadata', {})
            )
            
            logger.info(f"Successfully processed {task.file_name}: {len(processed_chunks)} chunks, {len(processed_tables)} tables")
            
        finally:
            # Cleanup temp file
            if temp_file_path and os.path.exists(temp_file_path):
                FileUtils.cleanup_temp_file(temp_file_path)

    async def _create_temp_copy(self, source_path: str) -> str:
        """
        Create temporary copy of PDF file for processing
        """
        loop = asyncio.get_event_loop()
        
        def _copy_file():
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            temp_path = temp_file.name
            temp_file.close()
            shutil.copy2(source_path, temp_path)
            return temp_path
        
        return await loop.run_in_executor(None, _copy_file)

    async def _upload_to_appwrite(self, file_path: str, file_name: str) -> str:
        """
        Upload file to Appwrite storage
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.appwrite_client.upload_file,
            file_path,
            file_name
        )

    async def _extract_pdf_content(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text and tables from PDF
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.pdf_processor.extract_text_and_tables,
            file_path
        )

    async def _generate_embeddings(self, pdf_results: Dict, metadata: Dict) -> tuple:
        """
        Generate embeddings for text chunks and tables
        """
        loop = asyncio.get_event_loop()
        
        # Process text chunks
        processed_chunks = []
        if pdf_results.get('text_chunks'):
            processed_chunks = await loop.run_in_executor(
                None,
                self.embedding_manager.process_document_chunks,
                pdf_results['text_chunks'],
                metadata
            )
        
        # Process tables
        processed_tables = []
        if pdf_results.get('tables'):
            processed_tables = await loop.run_in_executor(
                None,
                self.embedding_manager.process_tables,
                pdf_results['tables'],
                metadata
            )
        
        return processed_chunks, processed_tables

    async def _store_in_database(self, file_id: str, file_name: str, appwrite_file_id: str,
                                processed_chunks: List[Dict], processed_tables: List[Dict],
                                metadata: Dict, pdf_metadata: Dict) -> None:
        """
        Store all processed data in vector database
        """
        loop = asyncio.get_event_loop()
        
        # Store file metadata
        await loop.run_in_executor(
            None,
            self.vector_db.store_file_metadata,
            file_id, file_name, appwrite_file_id,
            metadata['level'], metadata['paper'],
            metadata.get('module'), metadata.get('chapter'), metadata.get('unit'),
            pdf_metadata.get('total_pages', 0), metadata.get('source_file')
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
                file_id, file_name, table['data'], table.get('html', ''),
                table['embedding'], table.get('context_before', ''),
                table.get('context_after', ''), table.get('page_number', 0), i,
                metadata['level'], metadata['paper'],
                metadata.get('module'), metadata.get('chapter'), metadata.get('unit')
            )

    def _print_final_stats(self) -> None:
        """
        Print final processing statistics
        """
        total_time = self.stats['end_time'] - self.stats['start_time']
        
        logger.info("="*60)
        logger.info("BATCH PROCESSING COMPLETED")
        logger.info("="*60)
        logger.info(f"Total files: {self.stats['total_files']}")
        logger.info(f"Completed: {self.stats['completed']}")
        logger.info(f"Failed: {self.stats['failed']}")
        logger.info(f"Retries: {self.stats['retries']}")
        logger.info(f"Total time: {total_time:.1f}s")
        
        if self.stats['completed'] > 0:
            avg_time = total_time / self.stats['completed']
            logger.info(f"Average time per file: {avg_time:.1f}s")
        
        if self.failed_files:
            logger.info(f"Failed files: {[task.file_name for task in self.tasks if task.file_id in self.failed_files]}")
        
        success_rate = (self.stats['completed'] / self.stats['total_files']) * 100 if self.stats['total_files'] > 0 else 0
        logger.info(f"Success rate: {success_rate:.1f}%")
        logger.info("="*60)

async def main():
    """
    Main entry point for batch ingestion
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch ingest CA PDFs from folder structure')
    parser.add_argument('--ca-folder', default='ca', help='Path to CA folder (default: ca)')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers (default: 4)')
    parser.add_argument('--retries', type=int, default=3, help='Max retry attempts per file (default: 3)')
    parser.add_argument('--force', action='store_true', help='Process all files even if already completed')
    
    args = parser.parse_args()
    
    try:
        # Initialize batch ingestor
        ingestor = BatchIngestor(
            ca_folder_path=args.ca_folder,
            max_workers=args.workers,
            retry_attempts=args.retries
        )
        
        # Discover files
        tasks = ingestor.discover_files()
        
        if not tasks:
            logger.info("No PDF files found to process")
            return
        
        # Filter already processed files (unless --force)
        if not args.force:
            tasks = ingestor.check_existing_files(tasks)
        
        if not tasks:
            logger.info("All files already processed. Use --force to reprocess.")
            return
        
        # Process all files
        await ingestor.process_batch(tasks)
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
