import os
import hashlib
import tempfile
import shutil
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FileUtils:
    @staticmethod
    def generate_file_id(file_path: str) -> str:
        """Generate a unique file ID based on file content"""
        try:
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            
            file_name = os.path.basename(file_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            return f"{file_name}_{timestamp}_{file_hash[:8]}"
            
        except Exception as e:
            logger.error(f"Failed to generate file ID: {e}")
            return f"file_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename by removing invalid characters"""
        try:
            # Remove or replace invalid characters for file paths
            # Replace colons, slashes, and other problematic characters
            sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
            
            # Remove excessive underscores and spaces
            sanitized = re.sub(r'[_\s]+', '_', sanitized)
            
            # Ensure it doesn't start or end with dots or spaces
            sanitized = sanitized.strip('. _')
            
            # Ensure minimum length
            if not sanitized:
                sanitized = f"file_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            
            logger.info(f"Sanitized filename: '{filename}' -> '{sanitized}'")
            return sanitized
            
        except Exception as e:
            logger.error(f"Failed to sanitize filename: {e}")
            return f"file_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    
    @staticmethod
    def create_temp_file(content: bytes, suffix: str = ".pdf") -> str:
        """Create a temporary file with given content"""
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            temp_file.write(content)
            temp_file.close()
            return temp_file.name
            
        except Exception as e:
            logger.error(f"Failed to create temp file: {e}")
            raise
    
    @staticmethod
    def cleanup_temp_file(file_path: str):
        """Clean up temporary file"""
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                logger.info(f"Cleaned up temp file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp file: {e}")
    
    @staticmethod
    def validate_pdf_file(file_path: str) -> bool:
        """Validate if file is a valid PDF"""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(8)
                return header.startswith(b'%PDF-')
        except Exception as e:
            logger.error(f"PDF validation failed: {e}")
            return False
    
    @staticmethod
    def get_file_size(file_path: str) -> int:
        """Get file size in bytes"""
        try:
            return os.path.getsize(file_path)
        except Exception as e:
            logger.error(f"Failed to get file size: {e}")
            return 0

class TextUtils:
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        try:
            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Remove special characters but keep basic punctuation
            text = re.sub(r'[^\w\s\.\,\?\!\;\:\-\(\)\[\]\/\%\₹\$]', '', text)
            
            # Remove extra spaces
            text = text.strip()
            
            return text
            
        except Exception as e:
            logger.error(f"Text cleaning failed: {e}")
            return text
    
    @staticmethod
    def extract_financial_figures(text: str) -> List[Dict[str, str]]:
        """Extract financial figures from text"""
        try:
            figures = []
            
            # Pattern for Indian currency
            inr_pattern = r'₹\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
            inr_matches = re.finditer(inr_pattern, text)
            
            for match in inr_matches:
                figures.append({
                    'amount': match.group(0),
                    'value': match.group(1),
                    'currency': 'INR',
                    'position': match.start()
                })
            
            # Pattern for percentages
            percent_pattern = r'(\d+(?:\.\d+)?)\s*%'
            percent_matches = re.finditer(percent_pattern, text)
            
            for match in percent_matches:
                figures.append({
                    'amount': match.group(0),
                    'value': match.group(1),
                    'currency': 'percentage',
                    'position': match.start()
                })
            
            return figures
            
        except Exception as e:
            logger.error(f"Financial figure extraction failed: {e}")
            return []
    
    @staticmethod
    def extract_references(text: str) -> List[Dict[str, str]]:
        """Extract references to standards, sections, etc."""
        try:
            references = []
            
            # Ind AS references
            indas_pattern = r'Ind\s*AS\s*(\d+)'
            indas_matches = re.finditer(indas_pattern, text, re.IGNORECASE)
            
            for match in indas_matches:
                references.append({
                    'type': 'Ind AS',
                    'reference': match.group(0),
                    'number': match.group(1),
                    'position': match.start()
                })
            
            # Section references
            section_pattern = r'Section\s*(\d+(?:\([a-zA-Z0-9]+\))?)'
            section_matches = re.finditer(section_pattern, text, re.IGNORECASE)
            
            for match in section_matches:
                references.append({
                    'type': 'Section',
                    'reference': match.group(0),
                    'number': match.group(1),
                    'position': match.start()
                })
            
            return references
            
        except Exception as e:
            logger.error(f"Reference extraction failed: {e}")
            return []

class ValidationUtils:
    @staticmethod
    def validate_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean metadata"""
        try:
            validated = {}
            
            # Required fields
            required_fields = ['level', 'paper', 'file_name']
            for field in required_fields:
                if field in metadata and metadata[field]:
                    validated[field] = str(metadata[field]).strip()
                else:
                    raise ValueError(f"Missing required field: {field}")
            
            # Optional fields
            optional_fields = ['module', 'chapter', 'unit']
            for field in optional_fields:
                if field in metadata and metadata[field]:
                    validated[field] = str(metadata[field]).strip()
                else:
                    validated[field] = None
            
            # Numeric fields
            if 'total_pages' in metadata:
                try:
                    validated['total_pages'] = int(metadata['total_pages'])
                except (ValueError, TypeError):
                    validated['total_pages'] = 0
            
            return validated
            
        except Exception as e:
            logger.error(f"Metadata validation failed: {e}")
            raise
    
    @staticmethod
    def validate_ca_level(level: str) -> bool:
        """Validate CA level"""
        valid_levels = ['Foundation', 'Intermediate', 'Final']
        return level in valid_levels
    
    @staticmethod
    def validate_question(question: str) -> bool:
        """Validate user question"""
        if not question or not question.strip():
            return False
        
        # Minimum length check
        if len(question.strip()) < 5:
            return False
        
        # Maximum length check
        if len(question.strip()) > 1000:
            return False
        
        return True

class ProgressTracker:
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0
        self.step_descriptions = {}
    
    def update(self, step: int, description: str = ""):
        """Update progress"""
        self.current_step = step
        if description:
            self.step_descriptions[step] = description
        
        logger.info(f"Progress: {step}/{self.total_steps} - {description}")
    
    def get_progress_percentage(self) -> float:
        """Get progress as percentage"""
        if self.total_steps == 0:
            return 100.0
        
        return (self.current_step / self.total_steps) * 100
    
    def is_complete(self) -> bool:
        """Check if process is complete"""
        return self.current_step >= self.total_steps

class ResponseFormatter:
    @staticmethod
    def format_answer_response(answer_data: Dict[str, Any]) -> str:
        """Format answer response for display"""
        try:
            formatted_parts = []
            
            # Main answer
            formatted_parts.append(f"**Answer:**\n{answer_data['answer']}")
            
            # Confidence
            if answer_data.get('confidence'):
                confidence_percent = answer_data['confidence'] * 100
                formatted_parts.append(f"**Confidence:** {confidence_percent:.1f}%")
            
            # Sources
            if answer_data.get('sources'):
                sources = answer_data['sources']
                
                if sources.get('documents'):
                    formatted_parts.append("**Document Sources:**")
                    for i, doc in enumerate(sources['documents'][:3]):
                        formatted_parts.append(f"{i+1}. {doc['file_name']} (Level: {doc['level']}, Paper: {doc['paper']})")
                
                if sources.get('tables'):
                    formatted_parts.append("**Table Sources:**")
                    for i, table in enumerate(sources['tables'][:2]):
                        formatted_parts.append(f"{i+1}. {table['file_name']}, Page {table['page_number']} ({table['rows']}x{table['cols']} table)")
            
            return "\n\n".join(formatted_parts)
            
        except Exception as e:
            logger.error(f"Response formatting failed: {e}")
            return answer_data.get('answer', 'Error formatting response')
    
    @staticmethod
    def format_processing_status(status_data: Dict[str, Any]) -> str:
        """Format processing status for display"""
        try:
            status_parts = []
            
            status_parts.append(f"**File:** {status_data.get('file_name', 'Unknown')}")
            status_parts.append(f"**Status:** {status_data.get('status', 'Unknown')}")
            
            if status_data.get('progress'):
                status_parts.append(f"**Progress:** {status_data['progress']:.1f}%")
            
            if status_data.get('current_step'):
                status_parts.append(f"**Current Step:** {status_data['current_step']}")
            
            if status_data.get('error'):
                status_parts.append(f"**Error:** {status_data['error']}")
            
            return "\n".join(status_parts)
            
        except Exception as e:
            logger.error(f"Status formatting failed: {e}")
            return "Error formatting status"
