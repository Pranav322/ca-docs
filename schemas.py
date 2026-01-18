"""
Pydantic schemas for FastAPI request/response validation
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


# ============ Request Schemas ============

class QuestionRequest(BaseModel):
    """Request schema for asking a question"""
    question: str = Field(..., min_length=5, max_length=1000, description="The question to ask")
    level: Optional[str] = Field(None, description="CA Level filter (Foundation, Intermediate, Final)")
    paper: Optional[str] = Field(None, description="Paper filter")
    module: Optional[str] = Field(None, description="Module filter")
    chapter: Optional[str] = Field(None, description="Chapter filter")
    unit: Optional[str] = Field(None, description="Unit filter")
    include_tables: bool = Field(True, description="Whether to include tables in search")


class DocumentUploadMetadata(BaseModel):
    """Metadata for document upload"""
    level: str = Field(..., description="CA Level (Foundation, Intermediate, Final)")
    paper: str = Field(..., description="Paper name")
    module: Optional[str] = Field(None, description="Module name")
    chapter: Optional[str] = Field(None, description="Chapter name")
    unit: Optional[str] = Field(None, description="Unit name")
    description: Optional[str] = Field(None, description="Optional description")
    tags: Optional[str] = Field(None, description="Comma-separated tags")


# ============ Response Schemas ============

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    version: str = "1.0.0"


class SourceDocument(BaseModel):
    """Document source in answer response"""
    file_name: str
    level: str
    paper: str
    chapter: Optional[str] = None
    similarity: float
    snippet: str


class SourceTable(BaseModel):
    """Table source in answer response"""
    file_name: str
    page_number: int
    level: str
    paper: str
    rows: int
    cols: int
    similarity: float
    context: Optional[str] = None


class AnswerSources(BaseModel):
    """Sources for an answer"""
    documents: List[SourceDocument] = []
    tables: List[SourceTable] = []


class AnswerMetadata(BaseModel):
    """Metadata about the answer"""
    documents_found: int = 0
    tables_found: int = 0
    processing_time_ms: Optional[float] = None


class QuestionResponse(BaseModel):
    """Response schema for question answering"""
    answer: str
    confidence: float
    sources: AnswerSources
    metadata: AnswerMetadata
    suggestions: List[str] = []


class FileMetadata(BaseModel):
    """File metadata response"""
    file_id: str
    file_name: str
    level: str
    paper: str
    module: Optional[str] = None
    chapter: Optional[str] = None
    unit: Optional[str] = None
    total_pages: int = 0
    processing_status: str
    upload_date: Optional[datetime] = None


class FileListResponse(BaseModel):
    """Response for file listing"""
    files: List[FileMetadata]
    total: int


class FileStatsResponse(BaseModel):
    """Response for file statistics"""
    total_files: int
    completed: int
    pending: int
    failed: int
    total_pages: int
    total_chunks: int = 0
    total_tables: int = 0


class DocumentUploadResponse(BaseModel):
    """Response for document upload"""
    success: bool
    file_id: Optional[str] = None
    file_name: str
    message: str
    pages: int = 0
    chunks: int = 0
    tables: int = 0


class CurriculumItem(BaseModel):
    """A curriculum item (paper, module, chapter, or unit)"""
    name: str
    path: str
    children: Optional[List["CurriculumItem"]] = None


class CurriculumResponse(BaseModel):
    """Full curriculum hierarchy response"""
    levels: Dict[str, Any]


class ErrorResponse(BaseModel):
    """Error response"""
    detail: str
    error_code: Optional[str] = None
