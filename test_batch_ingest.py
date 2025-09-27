#!/usr/bin/env python3
"""
Test script for batch ingestion functionality
"""

import asyncio
import os
import logging
import signal
from pathlib import Path
from batch_ingest import BatchIngestor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    global shutdown_requested
    logger.info(f"Received signal {signum}. Requesting graceful shutdown...")
    shutdown_requested = True

async def test_batch_ingest():
    """
    Test batch ingestion with a small subset of files
    """
    try:
        # Check if ca folder exists
        ca_folder = Path("ca")
        if not ca_folder.exists():
            logger.error("CA folder not found. Please ensure 'ca' folder exists with PDF files.")
            return
        
        # Initialize batch ingestor with fewer workers for testing
        ingestor = BatchIngestor(
            ca_folder_path="ca",
            max_workers=2,  # Use fewer workers for testing
            retry_attempts=2
        )
        
        # Test file discovery
        logger.info("=== Testing File Discovery ===")
        tasks = ingestor.discover_files()
        
        if not tasks:
            logger.info("No PDF files found in ca/ folder")
            return
        
        logger.info(f"Found {len(tasks)} PDF files:")
        for i, task in enumerate(tasks[:5]):  # Show first 5 files
            logger.info(f"  {i+1}. {task.file_name} -> {task.metadata}")
        
        if len(tasks) > 5:
            logger.info(f"  ... and {len(tasks) - 5} more files")
        
        # Test metadata validation
        logger.info("=== Testing Metadata Validation ===")
        valid_files = [task for task in tasks if ingestor._validate_metadata(task.metadata)]
        logger.info(f"Valid files: {len(valid_files)}/{len(tasks)}")
        
        # Check for existing files
        logger.info("=== Testing Existing Files Check ===")
        new_tasks = ingestor.check_existing_files(tasks)
        logger.info(f"New files to process: {len(new_tasks)}")
        
        # Ask user if they want to continue with actual processing
        if new_tasks:
            logger.info(f"Would process {len(new_tasks)} files with these settings:")
            logger.info(f"  Max workers: {ingestor.max_workers}")
            logger.info(f"  Retry attempts: {ingestor.retry_attempts}")
            
            # Check for shutdown request before prompting
            if shutdown_requested:
                logger.info("Shutdown requested before processing")
                return
            
            # Process just one file for testing if user confirms
            response = input("Process one test file? (y/N): ").strip().lower()
            if response in ['y', 'yes']:
                logger.info("=== Testing Single File Processing ===")
                test_task = new_tasks[0]
                logger.info(f"Processing test file: {test_task.file_name}")
                
                # Process just one file
                try:
                    await ingestor.process_batch([test_task])
                    logger.info("Single file processing completed successfully")
                except KeyboardInterrupt:
                    logger.info("Processing interrupted by user (Ctrl+C)")
                    return
                except Exception as e:
                    logger.error(f"Processing failed: {e}")
                    raise
                finally:
                    # Ensure cleanup of resources
                    try:
                        if hasattr(ingestor, 'vector_db') and hasattr(ingestor.vector_db, 'close'):
                            await ingestor.vector_db.close()
                        elif hasattr(ingestor, 'vector_db') and hasattr(ingestor.vector_db.pool, 'close'):
                            await ingestor.vector_db.pool.close()
                    except Exception as cleanup_error:
                        logger.warning(f"Cleanup error (non-critical): {cleanup_error}")
            else:
                logger.info("Skipping actual processing")
        else:
            logger.info("No new files to process")
        
        logger.info("=== Test Complete ===")
        
    except KeyboardInterrupt:
        logger.info("Test interrupted by user (Ctrl+C)")
        return
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise
    finally:
        # Add a small delay to ensure all async operations complete
        await asyncio.sleep(0.1)
        logger.info("Test cleanup completed")

if __name__ == "__main__":
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        asyncio.run(test_batch_ingest())
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.error(f"Application failed: {e}")
        raise
