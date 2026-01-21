import os
import asyncio
import uuid
import tempfile
import shutil
import logging
import time
import re
from typing import Dict, Any, List, Optional
from pathlib import Path
import concurrent.futures
from dataclasses import dataclass
from enum import Enum
from threading import Lock

# Import existing components (config.py now contains the DNS patch)
from database import VectorDatabase
from pdf_processor import PDFProcessor
from embeddings import EmbeddingManager
from utils import FileUtils, ValidationUtils
from config import CHUNK_SIZE, CHUNK_OVERLAP
from classifier import ContentClassifier, generate_node_id, ContentType

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("batch_ingest.log"), logging.StreamHandler()],
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
    def __init__(
        self, ca_folder_path: str = "ca", max_workers: int = 4, retry_attempts: int = 3
    ):
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
        logger.info("=== Initializing VectorDatabase ===")
        self.vector_db = VectorDatabase()
        logger.info("=== VectorDatabase initialized ===")
        logger.info("=== Initializing PDFProcessor ===")
        self.pdf_processor = PDFProcessor()
        logger.info("=== PDFProcessor initialized ===")
        logger.info("=== Initializing EmbeddingManager ===")
        self.embedding_manager = EmbeddingManager()
        logger.info("=== EmbeddingManager initialized ===")
        logger.info("=== Initializing ContentClassifier ===")
        self.classifier = ContentClassifier(
            use_llm_fallback=False
        )  # Smart regex only - 10x faster!
        logger.info("=== ContentClassifier initialized ===")

        # Processing stats
        self.stats = {
            "total_files": 0,
            "completed": 0,
            "failed": 0,
            "retries": 0,
            "start_time": 0,
            "end_time": 0,
        }

        # Task tracking
        self.tasks: List[FileTask] = []
        self.completed_files = set()
        self.failed_files = set()
        self.progress_lock = Lock()
        self.last_progress_update = 0

        logger.info(
            f"Batch Ingestor initialized with {max_workers} workers, max {retry_attempts} retries"
        )

    def _update_progress(self) -> None:
        """Update and display progress information (thread-safe)"""
        with self.progress_lock:
            current_time = time.time()
            if (
                current_time - self.last_progress_update < 30
                and self.stats["completed"] + self.stats["failed"]
                < self.stats["total_files"]
            ):
                return

            completed = self.stats["completed"]
            failed = self.stats["failed"]
            total = self.stats["total_files"]
            remaining = total - completed - failed

            if total > 0:
                progress_pct = ((completed + failed) / total) * 100
                elapsed_time = current_time - self.stats["start_time"]

                if completed > 0:
                    avg_time_per_file = elapsed_time / (completed + failed)
                    eta_seconds = avg_time_per_file * remaining
                    eta_hours = eta_seconds / 3600
                    eta_display = (
                        f"{eta_hours:.1f}h"
                        if eta_hours >= 1
                        else f"{eta_seconds / 60:.0f}m"
                    )
                else:
                    eta_display = "calculating..."

                logger.info(
                    f"📊 PROGRESS: {completed}/{total} completed ({progress_pct:.1f}%) | {failed} failed | {remaining} remaining | ETA: {eta_display}"
                )

            self.last_progress_update = current_time

    def discover_files(self) -> List[FileTask]:
        """Discover all PDF files in ca folder and create tasks with inferred metadata"""
        logger.info(f"Discovering PDF files in {self.ca_folder_path}")

        if not self.ca_folder_path.exists():
            raise FileNotFoundError(f"CA folder not found: {self.ca_folder_path}")

        tasks = []

        for pdf_file in self.ca_folder_path.rglob("*.pdf"):
            try:
                rel_path = pdf_file.relative_to(self.ca_folder_path)
                path_parts = rel_path.parts

                metadata = self._build_metadata(path_parts, str(rel_path))

                if not self._validate_metadata(metadata):
                    logger.warning(
                        f"Skipping {pdf_file} - invalid metadata: {metadata}"
                    )
                    continue

                # Use stable file ID based on content hash (avoids duplicates)
                file_id = FileUtils.generate_file_id(str(pdf_file))

                task = FileTask(
                    file_path=str(pdf_file),
                    file_name=pdf_file.name,
                    file_id=file_id,
                    metadata=metadata,
                    max_attempts=self.retry_attempts,
                )

                tasks.append(task)
                logger.debug(f"Added task: {pdf_file.name} -> {metadata}")

            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {e}")
                continue

        logger.info(f"Discovered {len(tasks)} PDF files for processing")
        return tasks

    def _build_metadata(self, path_parts: tuple, file_path: str) -> Dict[str, Any]:
        """Build metadata from path parts"""
        metadata = {
            "level": None,
            "paper": None,
            "module": None,
            "chapter": None,
            "unit": None,
            "source_file": file_path,
            "source_type": "ICAI_Module",
            "applicable_attempts": ["May_2026"],
        }

        for part in path_parts:
            part_lower = part.lower()

            if "foundation" in part_lower:
                metadata["level"] = "Foundation"
            elif "intermediate" in part_lower:
                metadata["level"] = "Intermediate"
            elif "final" in part_lower:
                metadata["level"] = "Final"

            if "paper" in part_lower:
                metadata["paper"] = part

            if "module" in part_lower:
                metadata["module"] = part

            if "chapter" in part_lower:
                metadata["chapter"] = part

            if "unit" in part_lower:
                metadata["unit"] = part

        # Generate stable node_id
        metadata["node_id"] = generate_node_id(file_path)

        return metadata

    def _validate_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Validate that required metadata fields are present"""
        return metadata.get("level") is not None and metadata.get("paper") is not None

    def check_existing_files(self, tasks: List[FileTask]) -> List[FileTask]:
        """Filter out already processed files"""
        existing_files = self.vector_db.get_file_metadata()

        completed_paths = set()
        for file_meta in existing_files:
            if file_meta.get("processing_status") == "completed":
                source = file_meta.get("source_file", "")
                if source:
                    completed_paths.add(source)

        pending_tasks = []
        for task in tasks:
            source_file = task.metadata.get("source_file", "")
            if source_file not in completed_paths:
                pending_tasks.append(task)
            else:
                logger.debug(f"Skipping already processed: {task.file_name}")

        skipped = len(tasks) - len(pending_tasks)
        if skipped > 0:
            logger.info(f"Skipped {skipped} already processed files")

        return pending_tasks

    async def process_batch(self, tasks: List[FileTask]) -> None:
        """Process all tasks with concurrency control"""
        self.tasks = tasks
        self.stats["total_files"] = len(tasks)
        self.stats["start_time"] = time.time()

        logger.info(
            f"🚀 Starting batch processing of {len(tasks)} files with {self.max_workers} workers"
        )

        semaphore = asyncio.Semaphore(self.max_workers)

        async def process_with_semaphore(task: FileTask):
            async with semaphore:
                await self._process_task(task)

        await asyncio.gather(*[process_with_semaphore(task) for task in tasks])

        self.stats["end_time"] = time.time()
        self._print_final_stats()

    async def _process_task(self, task: FileTask) -> None:
        """Process a single task with retry logic"""
        while task.attempt < task.max_attempts:
            task.attempt += 1
            task.status = ProcessingStatus.PROCESSING

            start_time = time.time()

            try:
                logger.info(
                    f"Processing [{task.attempt}/{task.max_attempts}]: {task.file_name}"
                )

                # IMMEDIATE UPDATE: Mark as processing in DB so dashboard sees it
                # We use store_file_metadata to ensure the record exists
                self.vector_db.store_file_metadata(
                    file_id=task.file_id,
                    file_name=task.file_name,
                    file_path=task.metadata.get("source_file", task.file_path),
                    level=task.metadata["level"],
                    paper=task.metadata["paper"],
                    module=task.metadata.get("module"),
                    chapter=task.metadata.get("chapter"),
                    unit=task.metadata.get("unit"),
                    total_pages=0,
                )
                self.vector_db.update_processing_status(task.file_id, "processing")

                await self._process_single_file(task)

                task.status = ProcessingStatus.COMPLETED
                task.processing_time = time.time() - start_time

                with self.progress_lock:
                    self.stats["completed"] += 1
                    self.completed_files.add(task.file_id)

                self._update_progress()
                return

            except Exception as e:
                task.error = str(e)
                logger.error(
                    f"Error processing {task.file_name} (attempt {task.attempt}): {e}"
                )

                if task.attempt < task.max_attempts:
                    task.status = ProcessingStatus.RETRYING
                    with self.progress_lock:
                        self.stats["retries"] += 1
                    await asyncio.sleep(2**task.attempt)
                else:
                    task.status = ProcessingStatus.FAILED
                    # Update DB status to failed
                    try:
                        self.vector_db.update_processing_status(task.file_id, "failed")
                    except Exception as db_err:
                        logger.error(
                            f"Failed to update status to failed for {task.file_name}: {db_err}"
                        )

                    with self.progress_lock:
                        self.stats["failed"] += 1
                        self.failed_files.add(task.file_id)

    async def _process_single_file(self, task: FileTask) -> None:
        """Process a single PDF file through the complete pipeline"""
        temp_file_path = None

        try:
            if not FileUtils.validate_pdf_file(task.file_path):
                raise Exception(f"Invalid PDF file: {task.file_path}")

            temp_file_path = await self._create_temp_copy(task.file_path)

            pdf_results = await self._extract_pdf_content(temp_file_path)

            # Classify chunks using hybrid classifier
            classified_chunks = await self._classify_and_generate_embeddings(
                pdf_results, task.metadata
            )

            await self._store_in_database(
                task.file_id,
                task.file_name,
                task.metadata.get("source_file", task.file_path),
                classified_chunks["processed_chunks"],
                classified_chunks["processed_tables"],
                task.metadata,
                pdf_results.get("metadata", {}),
            )

            self.vector_db.update_processing_status(task.file_id, "completed")

            logger.info(
                f"Successfully processed {task.file_name}: {len(classified_chunks['processed_chunks'])} chunks, {len(classified_chunks['processed_tables'])} tables"
            )

        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                FileUtils.cleanup_temp_file(temp_file_path)

    async def _create_temp_copy(self, source_path: str) -> str:
        """Create temporary copy of PDF file for processing"""
        loop = asyncio.get_event_loop()

        def _copy_file():
            temp_dir = os.path.join(os.getcwd(), "temp_processing")
            os.makedirs(temp_dir, exist_ok=True)

            temp_file = tempfile.NamedTemporaryFile(
                delete=False, suffix=".pdf", dir=temp_dir
            )
            temp_path = temp_file.name
            temp_file.close()
            shutil.copy2(source_path, temp_path)
            return temp_path

        return await loop.run_in_executor(None, _copy_file)

    async def _extract_pdf_content(self, file_path: str) -> Dict[str, Any]:
        """Extract text and tables from PDF"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.pdf_processor.extract_text_and_tables, file_path
        )

    async def _classify_and_generate_embeddings(
        self, pdf_results: Dict, metadata: Dict
    ) -> Dict:
        """Classify content types and generate embeddings"""
        loop = asyncio.get_event_loop()

        processed_chunks = []
        processed_tables = []

        # Process text chunks with classification
        if pdf_results.get("text_chunks"):
            chunks = pdf_results["text_chunks"]

            # Classify each chunk
            for chunk in chunks:
                content = (
                    chunk.get("content", chunk) if isinstance(chunk, dict) else chunk
                )
                content_type = self.classifier.classify(content)

                # Get embedding via azure_embeddings inner object
                embedding = await loop.run_in_executor(
                    None, self.embedding_manager.azure_embeddings.get_embedding, content
                )

                processed_chunks.append(
                    {
                        "content": content,
                        "embedding": embedding,
                        "content_type": content_type.value,
                        "metadata": chunk.get("metadata", {})
                        if isinstance(chunk, dict)
                        else {},
                    }
                )

        # Process tables (pre-classified as TABLE)
        if pdf_results.get("tables"):
            processed_tables = await loop.run_in_executor(
                None,
                self.embedding_manager.process_tables,
                pdf_results["tables"],
                metadata,
            )

        return {
            "processed_chunks": processed_chunks,
            "processed_tables": processed_tables,
        }

    async def _store_in_database(
        self,
        file_id: str,
        file_name: str,
        file_path: str,
        processed_chunks: List[Dict],
        processed_tables: List[Dict],
        metadata: Dict,
        pdf_metadata: Dict,
    ) -> None:
        """Store all processed data in vector database"""
        loop = asyncio.get_event_loop()

        # Store file metadata
        await loop.run_in_executor(
            None,
            self.vector_db.store_file_metadata,
            file_id,
            file_name,
            file_path,
            metadata["level"],
            metadata["paper"],
            metadata.get("module"),
            metadata.get("chapter"),
            metadata.get("unit"),
            pdf_metadata.get("total_pages", 0),
        )

        # Store document chunks with new fields
        for i, chunk in enumerate(processed_chunks):
            await loop.run_in_executor(
                None,
                lambda c=chunk, idx=i: self.vector_db.store_document_chunks_batch(
                    [
                        {
                            "file_id": file_id,
                            "file_name": file_name,
                            "content": c["content"],
                            "embedding": c["embedding"],
                            "metadata": c.get("metadata", {}),
                            "chunk_index": idx,
                            "level": metadata["level"],
                            "paper": metadata["paper"],
                            "module": metadata.get("module"),
                            "chapter": metadata.get("chapter"),
                            "unit": metadata.get("unit"),
                            "content_type": c.get("content_type", "text"),
                            "source_type": metadata.get("source_type", "ICAI_Module"),
                            "applicable_attempts": metadata.get(
                                "applicable_attempts", ["May_2026"]
                            ),
                            "node_id": metadata.get("node_id"),
                        }
                    ]
                ),
            )

        # Store tables
        for i, table in enumerate(processed_tables):
            await loop.run_in_executor(
                None,
                self.vector_db.store_table,
                file_id,
                file_name,
                table["data"],
                table.get("html", ""),
                table["embedding"],
                table.get("context_before", ""),
                table.get("context_after", ""),
                table.get("page_number", 0),
                i,
                metadata["level"],
                metadata["paper"],
                metadata.get("module"),
                metadata.get("chapter"),
                metadata.get("unit"),
            )

    def _print_final_stats(self) -> None:
        """Print final processing statistics"""
        total_time = self.stats["end_time"] - self.stats["start_time"]
        total_hours = total_time / 3600

        logger.info("🏆 " + "=" * 58)
        logger.info("🏆 BATCH PROCESSING COMPLETED")
        logger.info("🏆 " + "=" * 58)
        logger.info(f"📊 Total files: {self.stats['total_files']}")
        logger.info(f"✅ Completed: {self.stats['completed']}")
        logger.info(f"❌ Failed: {self.stats['failed']}")
        logger.info(f"🔁 Retries: {self.stats['retries']}")
        logger.info(f"⏱️ Total time: {total_hours:.2f}h ({total_time:.1f}s)")

        if self.stats["completed"] > 0:
            avg_time = total_time / self.stats["completed"]
            avg_minutes = avg_time / 60
            logger.info(
                f"⏱️ Average time per file: {avg_minutes:.1f}m ({avg_time:.1f}s)"
            )

        if self.failed_files:
            failed_names = [
                task.file_name
                for task in self.tasks
                if task.file_id in self.failed_files
            ]
            logger.info(
                f"⚠️ Failed files ({len(failed_names)}): {failed_names[:5]}{'...' if len(failed_names) > 5 else ''}"
            )

        success_rate = (
            (self.stats["completed"] / self.stats["total_files"]) * 100
            if self.stats["total_files"] > 0
            else 0
        )
        logger.info(f"🎯 Success rate: {success_rate:.1f}%")
        logger.info("🏆 " + "=" * 58)

        self._cleanup_temp_processing_dir()

    def _cleanup_temp_processing_dir(self) -> None:
        """Clean up the temp_processing directory after batch processing"""
        try:
            temp_dir = os.path.join(os.getcwd(), "temp_processing")
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temp processing directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp processing directory: {e}")


async def main():
    """Main entry point for batch ingestion"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch ingest CA PDFs from folder structure"
    )
    parser.add_argument(
        "--ca-folder", default="ca", help="Path to CA folder (default: ca)"
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Max retry attempts per file (default: 3)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Process all files even if already completed",
    )

    args = parser.parse_args()

    try:
        ingestor = BatchIngestor(
            ca_folder_path=args.ca_folder,
            max_workers=args.workers,
            retry_attempts=args.retries,
        )

        tasks = ingestor.discover_files()

        if not tasks:
            logger.info("No PDF files found to process")
            return

        if not args.force:
            tasks = ingestor.check_existing_files(tasks)

        if not tasks:
            logger.info("All files already processed. Use --force to reprocess.")
            return

        await ingestor.process_batch(tasks)

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
