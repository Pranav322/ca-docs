# Automated CA PDF Batch Processing

This document explains how to use the automated batch processing system to ingest all your CA PDFs efficiently with parallel processing and retries.

## Overview

The batch processing system automatically:
- 🔍 **Discovers** all PDF files in your `ca/` folder
- 🏷️ **Infers** curriculum metadata from folder structure (Level → Paper → Module → Chapter → Unit)
- ⚡ **Processes** files in parallel with multiple workers
- 🔄 **Retries** failed files with exponential backoff
- 💾 **Stores** everything consistently in the vector database
- 📊 **Tracks** progress and provides detailed statistics

## Folder Structure

Organize your PDFs like this for automatic metadata inference:

```
ca/
├── Foundation/
│   ├── Paper 1/
│   │   ├── Module A/
│   │   │   ├── Chapter 1.pdf
│   │   │   └── Unit 1.pdf
│   │   └── some-document.pdf
│   ├── Paper 2/
│   └── Paper 3/
├── Intermediate/
│   ├── Group I Paper 1/
│   ├── Group I Paper 2/
│   └── Group II Paper 5/
└── Final/
    ├── Group I Paper 1/
    └── Group II Paper 5/
```

## Usage

### 1. Explore Your CA Folder

First, check your folder structure and see how metadata will be inferred:

```bash
uv run python explore_ca_folder.py
```

This shows:
- 📁 Folder structure
- 🏷️ Inferred metadata for each file
- ✅ Validation results
- 📊 Summary statistics

### 2. Test with One File

Test the processing pipeline with a single file:

```bash
uv run python test_batch_ingest.py
```

This allows you to:
- 🔍 Verify file discovery works
- ✅ Check metadata validation
- 🧪 Process one test file safely

### 3. Full Batch Processing

Process all your CA PDFs automatically:

```bash
# Basic usage (4 workers, 3 retries)
uv run python batch_ingest.py

# Custom settings
uv run python batch_ingest.py --workers 6 --retries 5

# Force reprocess all files (even completed ones)
uv run python batch_ingest.py --force

# Process specific folder
uv run python batch_ingest.py --ca-folder /path/to/your/ca/folder
```

### Command Line Options

```bash
uv run python batch_ingest.py --help
```

Options:
- `--ca-folder PATH`: Path to CA folder (default: `ca`)
- `--workers N`: Number of parallel workers (default: 4)  
- `--retries N`: Max retry attempts per file (default: 3)
- `--force`: Process all files even if already completed

## Processing Pipeline

Each PDF goes through this pipeline:

1. **📄 File Validation** - Verify PDF is valid
2. **☁️ Upload to Appwrite** - Store in cloud storage
3. **📊 PDF Processing** - Extract text and tables using multiple methods
4. **🧠 Generate Embeddings** - Create vector embeddings for search
5. **💾 Store in Database** - Save to PostgreSQL with pgvector

## Features

### ⚡ Parallel Processing
- Multiple worker processes handle files concurrently
- Configurable worker count (default: 4)
- Efficient resource utilization

### 🔄 Robust Retry Logic
- Automatic retries for failed files
- Exponential backoff (2^attempt seconds)
- Configurable max attempts (default: 3)

### 📊 Progress Tracking
- Real-time logging of all operations
- Detailed error reporting
- Final statistics summary

### 🔍 Smart Duplicate Detection
- Automatically skips already processed files
- Compares by file path in database
- Use `--force` to override

### 🏷️ Automatic Metadata Inference
- Extracts Level, Paper, Module, Chapter, Unit from folder structure
- Validates against CA curriculum (Foundation/Intermediate/Final)
- Consistent with existing manual upload system

## Monitoring

### Logs
- Real-time logs in terminal
- Detailed logs saved to `batch_ingest.log`
- Separate logs for each processing step

### Progress Indicators
- Files discovered: X total
- Files to process: X new
- Processing: X/X (with retry counts)
- Results: X completed, X failed

### Final Statistics
```
============================================================
BATCH PROCESSING COMPLETED
============================================================
Total files: 150
Completed: 147
Failed: 3
Retries: 8
Total time: 1800.5s
Average time per file: 12.2s
Success rate: 98.0%
============================================================
```

## Error Handling

### Common Issues and Solutions

**❌ "Invalid PDF file"**
- Check file is not corrupted
- Ensure file has .pdf extension
- Verify file size > 0

**❌ "Invalid metadata"**
- Check folder structure matches expected hierarchy
- Ensure level is Foundation/Intermediate/Final
- At minimum needs Level and Paper

**❌ "Database connection failed"** 
- Check DATABASE_URL in .env file
- Ensure PostgreSQL is running
- Verify pgvector extension is installed

**❌ "Azure OpenAI API error"**
- Check AZURE_OPENAI_* variables in .env
- Verify API key is valid
- Check rate limits

### Recovery
- Failed files are automatically retried
- Check `batch_ingest.log` for detailed error messages
- Use `--force` to reprocess specific files
- Database handles partial failures gracefully

## Performance Tuning

### Optimal Settings
- **Small collections (<50 PDFs)**: `--workers 2`
- **Medium collections (50-200 PDFs)**: `--workers 4` (default)
- **Large collections (>200 PDFs)**: `--workers 6-8`

### Considerations
- More workers = faster processing but higher resource usage
- Azure OpenAI has rate limits (respect them)
- Database can handle concurrent connections well
- Monitor system memory usage

## Integration

The batch processor uses the same components as the Streamlit app:
- ✅ Same PDF processing pipeline
- ✅ Same embedding generation
- ✅ Same database schema
- ✅ Same metadata structure

Files processed via batch are immediately available in the web interface for Q&A.

## Next Steps

After batch processing completes:

1. **✅ Verify in Web App** - Start Streamlit and check File Management
2. **🔍 Test Questions** - Try asking questions about your materials  
3. **📊 Check Database** - Use database tools to verify data
4. **🔄 Regular Updates** - Run batch processing for new PDFs

## Troubleshooting

If you encounter issues:

1. Check the logs: `tail -f batch_ingest.log`
2. Test with one file: `python test_batch_ingest.py` 
3. Verify folder structure: `python explore_ca_folder.py`
4. Check environment variables in `.env`
5. Ensure all dependencies are installed: `uv sync`
