from appwrite.client import Client
from appwrite.services.storage import Storage
from appwrite.services.databases import Databases
from appwrite.services.account import Account
from appwrite.input_file import InputFile
from appwrite.id import ID
import os
import logging
from typing import Dict, Any, Optional, List
from config import APPWRITE_ENDPOINT, APPWRITE_PROJECT_ID, APPWRITE_API_KEY, APPWRITE_BUCKET_ID

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AppwriteClient:
    def __init__(self):
        self.client = Client()
        self.client.set_endpoint(APPWRITE_ENDPOINT)
        self.client.set_project(APPWRITE_PROJECT_ID)
        self.client.set_key(APPWRITE_API_KEY)
        
        self.storage = Storage(self.client)
        self.databases = Databases(self.client)
        
        self.bucket_id = APPWRITE_BUCKET_ID
        self.database_id = "ca_rag_metadata"
        self.collection_id = "file_metadata"
        
        self._ensure_setup()
    
    def _ensure_setup(self):
        """Ensure database and collections are set up"""
        try:
            # Try to get database, create if doesn't exist
            try:
                self.databases.get(self.database_id)
            except Exception:
                logger.info("Creating Appwrite database...")
                self.databases.create(
                    database_id=self.database_id,
                    name="CA RAG Metadata Database"
                )
            
            # Try to get collection, create if doesn't exist
            try:
                self.databases.get_collection(self.database_id, self.collection_id)
            except Exception:
                logger.info("Creating Appwrite collection...")
                self.databases.create_collection(
                    database_id=self.database_id,
                    collection_id=self.collection_id,
                    name="File Metadata Collection"
                )
                
                # Create attributes for the collection
                attributes = [
                    {"key": "file_id", "type": "string", "size": 255, "required": True},
                    {"key": "file_name", "type": "string", "size": 500, "required": True},
                    {"key": "level", "type": "string", "size": 100, "required": True},
                    {"key": "paper", "type": "string", "size": 200, "required": True},
                    {"key": "module", "type": "string", "size": 200, "required": False},
                    {"key": "chapter", "type": "string", "size": 200, "required": False},
                    {"key": "unit", "type": "string", "size": 200, "required": False},
                    {"key": "total_pages", "type": "integer", "required": False},
                    {"key": "processing_status", "type": "string", "size": 50, "required": False},
                ]
                
                for attr in attributes:
                    if attr["type"] == "string":
                        self.databases.create_string_attribute(
                            database_id=self.database_id,
                            collection_id=self.collection_id,
                            key=attr["key"],
                            size=attr["size"],
                            required=attr["required"]
                        )
                    elif attr["type"] == "integer":
                        self.databases.create_integer_attribute(
                            database_id=self.database_id,
                            collection_id=self.collection_id,
                            key=attr["key"],
                            required=attr["required"]
                        )
            
        except Exception as e:
            logger.warning(f"Appwrite setup warning: {e}")
    
    def upload_file(self, file_path: str, file_name: str) -> str:
        """Upload a PDF file to Appwrite storage"""
        try:
            # Create file input
            with open(file_path, 'rb') as file:
                file_input = InputFile.from_bytes(
                    file.read(),
                    filename=file_name
                )
            
            # Upload file
            result = self.storage.create_file(
                bucket_id=self.bucket_id,
                file_id=ID.unique(),
                file=file_input
            )
            
            logger.info(f"File uploaded successfully: {result['$id']}")
            return result['$id']
            
        except Exception as e:
            logger.error(f"Failed to upload file: {e}")
            raise
    
    def store_file_metadata(self, file_id: str, metadata: Dict[str, Any]) -> str:
        """Store file metadata in Appwrite database"""
        try:
            # Prepare document data
            document_data = {
                "file_id": file_id,
                "file_name": metadata.get("file_name", ""),
                "level": metadata.get("level", ""),
                "paper": metadata.get("paper", ""),
                "module": metadata.get("module", ""),
                "chapter": metadata.get("chapter", ""),
                "unit": metadata.get("unit", ""),
                "total_pages": metadata.get("total_pages", 0),
                "processing_status": metadata.get("processing_status", "pending")
            }
            
            # Create document
            result = self.databases.create_document(
                database_id=self.database_id,
                collection_id=self.collection_id,
                document_id=ID.unique(),
                data=document_data
            )
            
            logger.info(f"Metadata stored successfully: {result['$id']}")
            return result['$id']
            
        except Exception as e:
            logger.error(f"Failed to store metadata: {e}")
            raise
    
    def get_file_metadata(self, file_id: str = None) -> List[Dict[str, Any]]:
        """Retrieve file metadata from Appwrite"""
        try:
            if file_id:
                # Get specific file metadata
                results = self.databases.list_documents(
                    database_id=self.database_id,
                    collection_id=self.collection_id,
                    queries=[f'equal("file_id", "{file_id}")']
                )
            else:
                # Get all file metadata
                results = self.databases.list_documents(
                    database_id=self.database_id,
                    collection_id=self.collection_id
                )
            
            return [doc for doc in results['documents']]
            
        except Exception as e:
            logger.error(f"Failed to retrieve metadata: {e}")
            return []
    
    def update_processing_status(self, file_id: str, status: str) -> bool:
        """Update file processing status"""
        try:
            # First find the document
            results = self.databases.list_documents(
                database_id=self.database_id,
                collection_id=self.collection_id,
                queries=[f'equal("file_id", "{file_id}")']
            )
            
            if not results['documents']:
                logger.error(f"No document found for file_id: {file_id}")
                return False
            
            document_id = results['documents'][0]['$id']
            
            # Update the document
            self.databases.update_document(
                database_id=self.database_id,
                collection_id=self.collection_id,
                document_id=document_id,
                data={"processing_status": status}
            )
            
            logger.info(f"Processing status updated to {status} for file {file_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update processing status: {e}")
            return False
    
    def download_file(self, file_id: str, local_path: str) -> bool:
        """Download a file from Appwrite storage"""
        try:
            # Get file download
            file_data = self.storage.get_file_download(
                bucket_id=self.bucket_id,
                file_id=file_id
            )
            
            # Save to local path
            with open(local_path, 'wb') as file:
                file.write(file_data)
            
            logger.info(f"File downloaded successfully to {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download file: {e}")
            return False
    
    def delete_file(self, file_id: str) -> bool:
        """Delete a file from Appwrite storage"""
        try:
            self.storage.delete_file(
                bucket_id=self.bucket_id,
                file_id=file_id
            )
            
            logger.info(f"File deleted successfully: {file_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete file: {e}")
            return False
    
    def store_extracted_data(self, file_id: str, extracted_data: Dict[str, Any]) -> str:
        """Store extracted table and text data"""
        try:
            # Create a separate collection for extracted data if needed
            collection_id = "extracted_data"
            
            try:
                self.databases.get_collection(self.database_id, collection_id)
            except Exception:
                # Create collection for extracted data
                self.databases.create_collection(
                    database_id=self.database_id,
                    collection_id=collection_id,
                    name="Extracted Data Collection"
                )
                
                # Create attributes
                self.databases.create_string_attribute(
                    database_id=self.database_id,
                    collection_id=collection_id,
                    key="file_id",
                    size=255,
                    required=True
                )
                self.databases.create_string_attribute(
                    database_id=self.database_id,
                    collection_id=collection_id,
                    key="data_type",
                    size=50,
                    required=True
                )
                self.databases.create_string_attribute(
                    database_id=self.database_id,
                    collection_id=collection_id,
                    key="content",
                    size=10000,
                    required=True
                )
            
            # Store the extracted data
            result = self.databases.create_document(
                database_id=self.database_id,
                collection_id=collection_id,
                document_id=ID.unique(),
                data={
                    "file_id": file_id,
                    "data_type": extracted_data.get("type", "unknown"),
                    "content": str(extracted_data)[:10000]  # Truncate if too long
                }
            )
            
            logger.info(f"Extracted data stored successfully: {result['$id']}")
            return result['$id']
            
        except Exception as e:
            logger.error(f"Failed to store extracted data: {e}")
            raise
    
    def get_storage_usage(self) -> Dict[str, Any]:
        """Get storage usage statistics"""
        try:
            # List all files in bucket
            files = self.storage.list_files(bucket_id=self.bucket_id)
            
            total_size = sum(file.get('sizeOriginal', 0) for file in files['files'])
            
            return {
                'total_files': files['total'],
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2)
            }
            
        except Exception as e:
            logger.error(f"Failed to get storage usage: {e}")
            return {'total_files': 0, 'total_size_bytes': 0, 'total_size_mb': 0}
