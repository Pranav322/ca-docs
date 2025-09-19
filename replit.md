# Overview

This is a CA (Chartered Accountancy) RAG (Retrieval-Augmented Generation) Assistant built with Streamlit. The application allows users to upload CA study materials in PDF format, processes them to extract text and tables, stores the content in a vector database, and provides an intelligent Q&A interface. The system is specifically designed for CA course materials across Foundation, Intermediate, and Final levels, with structured paper and module organization.

# Recent Changes

**September 19, 2025**: Updated curriculum parsing system to use clean JSON structure instead of tree parsing for improved consistency and reliability. All hierarchical navigation now works with standardized naming from JSON file format.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Framework**: Streamlit web application with session state management
- **User Interface**: Single-page application with sidebar navigation and file upload capabilities
- **State Management**: Streamlit session state for component initialization, processing status, and chat history
- **File Handling**: Temporary file processing with automatic cleanup

## Backend Architecture
- **Modular Design**: Separated concerns across multiple specialized modules:
  - PDF processing for text and table extraction
  - Vector database operations for similarity search
  - Embedding generation and management
  - RAG pipeline for question answering
  - Table-specific processing for financial data
- **Processing Pipeline**: Multi-stage PDF processing using PyMuPDF, pdfplumber, and Camelot for comprehensive content extraction
- **RAG Implementation**: Context-aware retrieval with filtering by CA levels, papers, modules, and chapters

## Data Storage Architecture
- **Vector Database**: PostgreSQL with pgvector extension for storing document embeddings and metadata
- **Database Schema**: Separate tables for documents (text chunks) and tables (structured data) with rich metadata
- **File Storage**: Appwrite cloud storage for PDF file management with automatic database setup
- **Metadata Management**: Comprehensive tracking of file processing status, CA course structure mapping, and content relationships

## Authentication and Authorization
- **Service-based Authentication**: API key-based authentication for Azure OpenAI and Appwrite services
- **No User Authentication**: Simple single-user application without user management

## AI and ML Components
- **Embedding Model**: Azure OpenAI text-embedding-ada-002 for semantic search
- **Language Model**: Azure OpenAI GPT-4 for question answering
- **Batch Processing**: Optimized embedding generation with rate limiting and batch processing
- **Table-aware RAG**: Specialized processing for financial tables and numerical data common in CA materials

# External Dependencies

## AI Services
- **Azure OpenAI**: Primary AI service for embeddings (text-embedding-ada-002) and language model (GPT-4)
- **Configuration**: Endpoint, API key, and deployment names configurable via environment variables

## Database Services
- **PostgreSQL**: Primary database with pgvector extension for vector similarity search
- **Connection**: Full PostgreSQL connection parameters (host, port, database, user, password)

## Cloud Storage
- **Appwrite**: Cloud storage service for PDF file management
- **Configuration**: Endpoint, project ID, API key, and bucket ID for file storage
- **Auto-setup**: Automatic database and collection creation for metadata management

## PDF Processing Libraries
- **PyMuPDF (fitz)**: Primary PDF text extraction
- **pdfplumber**: Enhanced table detection and extraction
- **Camelot**: Advanced table extraction for complex layouts
- **tabula-py**: Alternative table extraction method
- **pytesseract**: OCR capabilities for image-based content

## Data Processing
- **pandas**: Data manipulation and table processing
- **numpy**: Numerical operations for embeddings and vector calculations
- **OpenCV**: Image processing for PDF content analysis

## Web Framework
- **Streamlit**: Complete web application framework with built-in state management and UI components

## Utility Libraries
- **psycopg2**: PostgreSQL database adapter
- **python-multipart**: File upload handling
- **Pillow**: Image processing support