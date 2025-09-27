# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

CA-Rag is a Retrieval-Augmented Generation (RAG) system designed for Chartered Accountant (CA) education materials. It's a Streamlit web application that processes PDF study materials, stores them in a PostgreSQL vector database, and provides intelligent question-answering capabilities with contextual retrieval.

## Development Commands

### Running the Application
```bash
# Start the Streamlit application
uv run streamlit run app.py
```

### Environment Setup
```bash
# Install dependencies using uv
uv sync

# Activate the virtual environment (if needed)
source .venv/bin/activate
```

### Testing
```bash
# Run individual test files (no pytest currently configured)
uv run python test_db_connection.py
uv run python test_initialization.py
uv run python test_async_queue.py

# Test batch ingestion with one file
uv run python test_batch_ingest.py
```

### Database Operations
```bash
# The application automatically initializes PostgreSQL with pgvector extension
# Database URL is configured via .env file: DATABASE_URL

# Test database connection
uv run python test_db_connection.py
```

### Automated Batch Processing
```bash
# Explore CA folder structure and see inferred metadata
uv run python explore_ca_folder.py

# Process all CA PDFs automatically with parallel processing
uv run python batch_ingest.py

# Custom batch processing with more workers
uv run python batch_ingest.py --workers 6 --retries 5

# Force reprocess all files
uv run python batch_ingest.py --force
```

## Architecture Overview

### Core Components

**Streamlit Frontend (`app.py`)**
- Multi-page interface for document upload, question answering, and file management
- Curriculum-aware filtering with both smart search and traditional hierarchical selection
- Real-time progress tracking for document processing

**Vector Database Layer (`database.py`)**
- PostgreSQL with pgvector extension for semantic search
- Three main tables: `documents` (text chunks), `tables` (extracted tables), `file_metadata`
- Supports CA curriculum hierarchy: Level → Paper → Module → Chapter → Unit

**RAG Pipeline (`rag_pipeline.py`)**
- Azure OpenAI integration for embeddings and LLM
- Hybrid retrieval: combines document chunks and structured table data
- Context-aware answer generation with confidence scoring and source citations

**PDF Processing (`pdf_processor.py`, `table_processor.py`)**
- Multi-format PDF parsing with table extraction
- Chunking strategy for optimal retrieval (1000 chars, 200 overlap)
- Table-specific processing with context preservation

**Curriculum Management (`curriculum_manager.py`, `curriculum_ui.py`)**
- Hierarchical CA syllabus structure (Foundation/Intermediate/Final levels)
- Smart search capabilities for quick curriculum navigation
- Metadata tagging system for precise content organization

### Data Flow

1. **Document Upload**: PDFs → curriculum tagging → Appwrite storage → processing queue
2. **Processing**: PDF parsing → text/table extraction → embedding generation → vector storage
3. **Query**: Question → embedding → similarity search → context assembly → LLM generation → formatted response

### Configuration

**Environment Variables** (`.env` file required):
- `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_KEY` - Azure OpenAI configuration
- `DATABASE_URL` - PostgreSQL connection string
- `APPWRITE_ENDPOINT`, `APPWRITE_PROJECT_ID`, `APPWRITE_API_KEY` - File storage

**CA Curriculum Structure** (`config.py`):
- Predefined hierarchy: Foundation (4 papers), Intermediate (8 papers), Final (8 papers)
- Each level supports Module → Chapter → Unit granularity

## Key Files

- `app.py` - Main Streamlit application entry point
- `rag_pipeline.py` - Core RAG logic and Azure OpenAI integration
- `database.py` - Vector database operations with pgvector
- `curriculum_manager.py` - CA syllabus structure management
- `pdf_processor.py` - Document parsing and chunking
- `config.py` - Configuration and CA curriculum definitions

## Development Notes

- Uses `uv` for dependency management (modern Python package manager)
- PostgreSQL with pgvector required for similarity search
- Azure OpenAI required for embeddings (text-embedding-ada-002) and LLM (GPT-4)
- Appwrite used for file storage and management
- CA-specific domain knowledge embedded in curriculum hierarchy

## Testing Strategy

Current tests focus on:
- Database connectivity and initialization
- Async queue processing functionality
- Component initialization verification

No formal testing framework configured - individual test files run directly with Python.
