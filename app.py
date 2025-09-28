import streamlit as st
import os
import tempfile
import uuid
from typing import Dict, Any, List, Optional
import logging
import time
from datetime import datetime
import asyncio
import nest_asyncio
from streamlit.runtime.scriptrunner import add_script_run_ctx
import threading

# Apply nest_asyncio
nest_asyncio.apply()

# Import our custom modules
from database import VectorDatabase
from pdf_processor import PDFProcessor
from table_processor import TableProcessor
from embeddings import EmbeddingManager
from rag_pipeline import RAGPipeline
from appwrite_client import AppwriteClient
from utils import FileUtils, ValidationUtils, ProgressTracker, ResponseFormatter
from config import CA_LEVELS, CA_PAPERS
from curriculum_ui import CurriculumSelector, render_curriculum_filter, render_smart_curriculum_selector
from curriculum_manager import curriculum_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.processing_status = {}
    st.session_state.uploaded_files = []
    st.session_state.chat_history = []

def run_async_task(async_func, *args, **kwargs):
    """Run an async function in the current event loop"""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(async_func(*args, **kwargs))

def initialize_components():
    """Initialize all components"""
    try:
        if not st.session_state.initialized:
            with st.spinner("Initializing CA RAG Assistant..."):
                st.session_state.vector_db = VectorDatabase()
                st.session_state.pdf_processor = PDFProcessor()
                st.session_state.table_processor = TableProcessor()
                st.session_state.embedding_manager = EmbeddingManager()
                st.session_state.rag_pipeline = RAGPipeline()
                st.session_state.appwrite_client = AppwriteClient()
                st.session_state.initialized = True
                st.success("CA RAG Assistant initialized successfully!")
        return True
    except Exception as e:
        st.error(f"Failed to initialize components: {str(e)}")
        logger.error(f"Initialization failed: {e}")
        return False

def render_sidebar():
    """Render sidebar with navigation and filters"""
    st.sidebar.title("🎓 CA RAG Assistant")
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Navigate to:",
        ["📚 Ask Questions", "📄 Upload Documents", "📊 File Management", "ℹ️ Help & Info"]
    )
    
    st.sidebar.markdown("---")
    
    # Global filters (for question answering) - Now using curriculum selector
    if page == "📚 Ask Questions":
        st.sidebar.subheader("🎯 Filter by Syllabus")
        
        # Choice between smart and traditional filtering
        filter_mode = st.sidebar.radio(
            "Filter Mode",
            ["🔍 Smart Search", "📋 Traditional"],
            key="sidebar_filter_mode",
            help="Smart Search for quick filtering or Traditional step-by-step"
        )
        
        # Use the appropriate filter component
        with st.sidebar:
            if filter_mode == "🔍 Smart Search":
                filters = render_smart_curriculum_selector(
                    prefix="sidebar_smart_filter",
                    title=""
                )
            else:
                filters = render_curriculum_filter(prefix="sidebar_filter", title="", show_clear=True)
            
            include_tables = st.checkbox("Include Tables", value=True, key="include_tables")
    
    return page

def render_file_upload():
    """Render file upload interface"""
    st.header("📄 Upload CA Study Materials")
    st.markdown("Upload multiple PDF files from CA syllabus with proper metadata tagging")

    # File upload - now supports multiple files
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload CA syllabus PDFs (Foundation, Intermediate, or Final level). Select multiple files to upload them as a batch."
    )

    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) selected")

        # Display files and their curriculum selectors
        st.subheader("🏷️ Curriculum Tagging for Each File")
        st.markdown("Select the curriculum hierarchy for each document individually")

        # Store file configurations
        file_configs = {}

        for i, uploaded_file in enumerate(uploaded_files):
            with st.expander(f"📄 {uploaded_file.name} ({uploaded_file.size} bytes)", expanded=(i==0)):
                st.markdown(f"**File {i+1}:** {uploaded_file.name}")

                # Choice between smart and traditional selector
                selector_type = st.radio(
                    "Selection Method",
                    ["🔍 Smart Search", "📋 Traditional Cascading"],
                    key=f"selector_type_{i}",
                    horizontal=True,
                    help="Smart Search: Type to find and auto-fill hierarchy. Traditional: Step-by-step selection."
                )

                if selector_type == "🔍 Smart Search":
                    # Use the new smart curriculum selector
                    selection = render_smart_curriculum_selector(
                        prefix=f"smart_file_{i}",
                        title=f"📖 Select Curriculum for {uploaded_file.name}"
                    )
                    
                    # Create a compatible selector object for validation
                    from curriculum_ui import SmartCurriculumSelector
                    file_selector = SmartCurriculumSelector(prefix=f"smart_file_{i}")
                else:
                    # Use traditional selector
                    file_selector = CurriculumSelector(prefix=f"file_{i}")
                    selection = file_selector.render_complete_selector(
                        title=f"📖 Select Curriculum for {uploaded_file.name}",
                        show_path=True,
                        columns=True
                    )
                    file_selector.render_selection_status()

                # Additional metadata for this file
                with st.expander("📋 Additional Information"):
                    description = st.text_area(
                        "Description (optional)",
                        key=f"description_{i}",
                        help=f"Optional description for {uploaded_file.name}"
                    )
                    tags = st.text_input(
                        "Tags (comma-separated)",
                        key=f"tags_{i}",
                        help=f"Optional tags for {uploaded_file.name}"
                    )

                # Store configuration
                file_configs[i] = {
                    'file': uploaded_file,
                    'selector': file_selector,
                    'description': description,
                    'tags': tags
                }

        # Batch upload button
        if st.button("🚀 Upload and Process All Files", type="primary", use_container_width=True):
            # Validate all selections
            invalid_files = []
            for i, config in file_configs.items():
                if not config['selector'].is_complete_selection():
                    invalid_files.append(uploaded_files[i].name)

            if invalid_files:
                st.error(f"Please complete curriculum selection for: {', '.join(invalid_files)}")
            else:
                # Process all files using the new async function
                run_async_task(process_multiple_files, file_configs)

async def process_multiple_files(file_configs: Dict[int, Dict], max_workers: int = 16):
    """Process multiple uploaded files concurrently with progress tracking"""
    total_files = len(file_configs)
    
    # Create progress containers
    overall_progress = st.progress(0)
    overall_status = st.empty()
    
    overall_status.text(f"🚀 Starting batch processing of {total_files} files with {max_workers} workers...")

    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_workers)
    
    # Prepare tasks
    tasks = []
    for i, (file_index, config) in enumerate(file_configs.items()):
        tasks.append(
            process_single_file_async(
                i, total_files, config, semaphore
            )
        )

    # Run tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    processed_files = [r for r in results if isinstance(r, dict) and r.get('success')]
    failed_files = [r for r in results if not (isinstance(r, dict) and r.get('success'))]

    # Show final summary
    overall_progress.progress(1.0)
    overall_status.text("🎉 Batch processing completed!")

    # Summary
    st.subheader("📊 Batch Processing Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Files", total_files)
    with col2:
        st.metric("Successfully Processed", len(processed_files))
    with col3:
        st.metric("Failed", len(failed_files))

    if processed_files:
        st.success("**Successfully processed files:**")
        for result in processed_files:
            st.write(f"• {result['file_name']} - {result['pages']} pages, {result['chunks']} chunks, {result['tables']} tables")

    if failed_files:
        st.error("**Failed files:**")
        for failed in failed_files:
            if isinstance(failed, dict):
                st.write(f"• {failed.get('file', 'Unknown')}: {failed.get('error', 'Unknown error')}")
            else:
                st.write(f"• An unexpected error occurred: {str(failed)}")

    # Add processed files to session state
    for result in processed_files:
        st.session_state.uploaded_files.append({
            'file_id': result['file_id'],
            'file_name': result['file_name'],
            'metadata': result['metadata'],
            'status': 'completed',
            'upload_time': time.time()
        })


async def process_single_file_async(file_index: int, total_files: int, config: Dict, semaphore: asyncio.Semaphore) -> Dict:
    """Asynchronous wrapper for processing a single file"""
    async with semaphore:
        loop = asyncio.get_event_loop()
        
        # Get data from config
        uploaded_file = config['file']
        selector = config['selector']
        description = config['description']
        tags = config['tags']
        final_selection = selector.get_current_selection()

        # Create a container for this file's progress
        container = st.empty()
        with container.container():
            st.markdown(f"### ⏳ Waiting to process: {uploaded_file.name}")

        try:
            # Get the current script run context
            ctx = st.runtime.scriptrunner.get_script_run_ctx()

            # Create a wrapper function that sets the context
            def context_wrapper():
                add_script_run_ctx(threading.current_thread(), ctx)
                return process_single_file_with_progress(
                    file_index,
                    total_files,
                    uploaded_file,
                    final_selection,
                    description,
                    tags,
                    container
                )

            # Run the synchronous processing function in a thread pool
            result = await loop.run_in_executor(
                None,  # Uses the default ThreadPoolExecutor
                context_wrapper
            )
            return result
        except Exception as e:
            logger.error(f"Error in async file processing for {uploaded_file.name}: {e}")
            return {'success': False, 'file': uploaded_file.name, 'error': str(e)}

def process_single_file_with_progress(file_index: int, total_files: int, uploaded_file, metadata: Dict[str, Any],
                                    description: str, tags: str, container) -> Dict[str, Any]:
    """Process a single file with progress tracking, returns result dict. This is a synchronous function."""
    
    # Update UI within the container
    with container.container():
        st.markdown(f"### 📄 Processing File {file_index + 1}/{total_files}: {uploaded_file.name}")
        progress_bar = st.progress(0)
        status_text = st.empty()

    try:
        # Create temporary file first
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # Generate unique file ID using the actual file path
        file_id = FileUtils.generate_file_id(tmp_file_path)

        # Add sanitized file name to metadata
        sanitized_filename = FileUtils.sanitize_filename(uploaded_file.name)
        metadata['file_name'] = sanitized_filename
        metadata['description'] = description or None
        metadata['tags'] = [tag.strip() for tag in tags.split(',') if tag.strip()] if tags else []

        # Validate metadata
        validated_metadata = ValidationUtils.validate_metadata(metadata)

        # Validate PDF
        if not FileUtils.validate_pdf_file(tmp_file_path):
            FileUtils.cleanup_temp_file(tmp_file_path)
            return {'success': False, 'error': 'Invalid PDF file'}

        # Step 1: Upload to Appwrite
        status_text.text("📤 Uploading file to storage...")
        progress_bar.progress(10)

        appwrite_file_id = st.session_state.appwrite_client.upload_file(tmp_file_path, uploaded_file.name)

        # Step 2: Store metadata
        status_text.text("💾 Storing file metadata...")
        progress_bar.progress(20)

        st.session_state.vector_db.store_file_metadata(
            file_id=file_id,
            file_name=sanitized_filename,
            appwrite_file_id=appwrite_file_id,
            level=validated_metadata['level'],
            paper=validated_metadata['paper'],
            module=validated_metadata.get('module'),
            chapter=validated_metadata.get('chapter'),
            unit=validated_metadata.get('unit')
        )

        # Step 3: Process PDF content
        status_text.text("🔍 Extracting text and tables...")
        progress_bar.progress(30)

        pdf_results = st.session_state.pdf_processor.extract_text_and_tables(tmp_file_path)

        # Step 4: Update page count
        total_pages = pdf_results['metadata']['total_pages']
        st.session_state.vector_db.store_file_metadata(
            file_id=file_id,
            file_name=sanitized_filename,
            appwrite_file_id=appwrite_file_id,
            level=validated_metadata['level'],
            paper=validated_metadata['paper'],
            module=validated_metadata.get('module'),
            chapter=validated_metadata.get('chapter'),
            unit=validated_metadata.get('unit'),
            total_pages=total_pages
        )

        # Step 5: Process text chunks
        if pdf_results['text_chunks']:
            status_text.text("✂️ Processing text chunks...")
            progress_bar.progress(50)

            # Create table-aware chunks
            tables_info = [{'extraction_method': t.get('extraction_method', ''), 'rows': t.get('rows', 0)}
                          for t in pdf_results['tables']]

            processed_chunks = []
            for chunk_data in pdf_results['text_chunks']:
                chunk_content = chunk_data['content']
                table_aware_chunks = st.session_state.table_processor.chunk_table_aware_text(
                    chunk_content, tables_info, chunk_size=1000, overlap=200
                )
                processed_chunks.extend(table_aware_chunks)

            # Step 6: Generate embeddings for chunks
            status_text.text("🧠 Generating embeddings for text...")
            progress_bar.progress(70)

            chunk_embeddings = st.session_state.embedding_manager.process_document_chunks(
                processed_chunks, validated_metadata
            )

            # Step 7: Store chunks in database
            status_text.text("💾 Storing text chunks...")
            progress_bar.progress(80)

            for i, chunk in enumerate(chunk_embeddings):
                st.session_state.vector_db.store_document_chunk(
                    file_id=file_id,
                    file_name=sanitized_filename,
                    content=chunk['content'],
                    embedding=chunk['embedding'],
                    metadata=chunk['metadata'],
                    chunk_index=i,
                    level=validated_metadata['level'],
                    paper=validated_metadata['paper'],
                    module=validated_metadata.get('module'),
                    chapter=validated_metadata.get('chapter'),
                    unit=validated_metadata.get('unit')
                )

        # Step 8: Process tables
        if pdf_results['tables']:
            status_text.text("📊 Processing tables...")
            progress_bar.progress(85)

            table_embeddings = st.session_state.embedding_manager.process_tables(
                pdf_results['tables'], validated_metadata
            )

            # Store tables in database
            for i, table in enumerate(table_embeddings):
                st.session_state.vector_db.store_table(
                    file_id=file_id,
                    file_name=sanitized_filename,
                    table_data=table['data'] if 'data' in table else {},
                    table_html=table.get('html', ''),
                    embedding=table['embedding'],
                    context_before=table.get('context_before', ''),
                    context_after=table.get('context_after', ''),
                    page_number=table.get('page_number', 0),
                    table_index=i,
                    level=validated_metadata['level'],
                    paper=validated_metadata['paper'],
                    module=validated_metadata.get('module'),
                    chapter=validated_metadata.get('chapter'),
                    unit=validated_metadata.get('unit')
                )

        # Step 9: Update processing status
        status_text.text("✅ Finalizing...")
        progress_bar.progress(95)

        st.session_state.vector_db.update_processing_status(file_id, "completed")

        progress_bar.progress(100)
        status_text.text("🎉 Processing completed successfully!")

        # Cleanup
        FileUtils.cleanup_temp_file(tmp_file_path)

        with container.container():
            st.success(f"✅ {uploaded_file.name} processed successfully!")

        return {
            'success': True,
            'file_id': file_id,
            'file_name': uploaded_file.name,
            'metadata': validated_metadata,
            'pages': total_pages,
            'chunks': len(pdf_results['text_chunks']),
            'tables': len(pdf_results['tables'])
        }

    except Exception as e:
        logger.error(f"File processing error for {uploaded_file.name}: {e}")
        with container.container():
            st.error(f"❌ Error processing {uploaded_file.name}: {e}")

        # Cleanup on error
        try:
            if 'tmp_file_path' in locals():
                FileUtils.cleanup_temp_file(tmp_file_path)
        except:
            pass

        return {'success': False, 'file': uploaded_file.name, 'error': str(e)}


def process_uploaded_file(uploaded_file, metadata: Dict[str, Any]):
    """Legacy function for single file processing - now delegates to the new function"""
    result = process_single_file_with_progress(
        uploaded_file, metadata, None, None,
        st.progress(0), st.empty()
    )

    if result['success']:
        st.success(f"""
        **File processed successfully!**
        - **File:** {uploaded_file.name}
        - **Level:** {result['metadata']['level']}
        - **Paper:** {result['metadata']['paper']}
        - **Total Pages:** {result['pages']}
        - **Text Chunks:** {result['chunks']}
        - **Tables Found:** {result['tables']}
        """)

        # Add to session state
        st.session_state.uploaded_files.append({
            'file_id': result['file_id'],
            'file_name': uploaded_file.name,
            'metadata': result['metadata'],
            'status': 'completed',
            'upload_time': time.time()
        })
    else:
        st.error(f"Error processing file: {result['error']}")

def render_question_interface():
    """Render question answering interface"""
    st.header("📚 Ask Questions about CA Syllabus")
    st.markdown("Ask any question about Chartered Accountancy topics and get comprehensive answers with references.")
    
    # Question input
    question = st.text_area(
        "Enter your question:",
        placeholder="e.g., What are the key components of a balance sheet as per Schedule III of Companies Act 2013?",
        height=100,
        help="Be specific for better results. You can ask about concepts, procedures, standards, or calculations."
    )
    
    # Question options
    col1, col2 = st.columns([3, 1])
    
    with col2:
        ask_button = st.button("🚀 Get Answer", type="primary")
        clear_button = st.button("🗑️ Clear History")
    
    if clear_button:
        st.session_state.chat_history = []
        st.rerun()
    
    if ask_button:
        if not question.strip():
            st.error("Please enter a question.")
            return
        
        if not ValidationUtils.validate_question(question):
            st.error("Please enter a valid question (5-1000 characters).")
            return
        
        # Get filters from the appropriate selector in sidebar
        filter_mode = st.session_state.get('sidebar_filter_mode', '📋 Traditional')
        
        if filter_mode == "🔍 Smart Search":
            level_filter = st.session_state.get('sidebar_smart_filter_level')
            paper_filter = st.session_state.get('sidebar_smart_filter_paper')
            module_filter = st.session_state.get('sidebar_smart_filter_module')
            chapter_filter = st.session_state.get('sidebar_smart_filter_chapter')
            unit_filter = st.session_state.get('sidebar_smart_filter_unit')
        else:
            level_filter = st.session_state.get('sidebar_filter_level')
            paper_filter = st.session_state.get('sidebar_filter_paper')
            module_filter = st.session_state.get('sidebar_filter_module')
            chapter_filter = st.session_state.get('sidebar_filter_chapter')
            unit_filter = st.session_state.get('sidebar_filter_unit')
        
        include_tables = st.session_state.get('include_tables', True)
        
        # Show processing
        with st.spinner("🔍 Searching knowledge base and generating answer..."):
            try:
                # Get answer from RAG pipeline
                answer_data = st.session_state.rag_pipeline.answer_question(
                    question=question,
                    level=level_filter,
                    paper=paper_filter,
                    module=module_filter,
                    chapter=chapter_filter,
                    unit=unit_filter,
                    include_tables=include_tables
                )
                
                # Add to chat history
                st.session_state.chat_history.append({
                    'question': question,
                    'answer_data': answer_data,
                    'timestamp': time.time(),
                    'filters': {
                        'level': level_filter,
                        'paper': paper_filter,
                        'module': module_filter,
                        'chapter': chapter_filter,
                        'unit': unit_filter
                    }
                })
                
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")
                logger.error(f"Question answering error: {e}")
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("💬 Q&A History")
        
        # Reverse to show latest first
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"Q: {chat['question'][:100]}{'...' if len(chat['question']) > 100 else ''}", expanded=(i==0)):
                render_answer_display(chat['answer_data'], chat['question'])
                
                # Show applied filters with curriculum hierarchy
                if any(chat['filters'].values()):
                    st.markdown("**🎯 Applied Filters:**")
                    # Create display path from hierarchy
                    path_parts = []
                    if chat['filters']['level']:
                        path_parts.append(chat['filters']['level'])
                    if chat['filters']['paper']:
                        path_parts.append(chat['filters']['paper'])
                    if chat['filters']['module']:
                        path_parts.append(chat['filters']['module'])
                    if chat['filters']['chapter']:
                        path_parts.append(chat['filters']['chapter'])
                    if chat['filters']['unit']:
                        path_parts.append(chat['filters']['unit'])
                    
                    if path_parts:
                        filter_path = " → ".join(path_parts)
                        st.caption(f"📍 {filter_path}")
                else:
                    st.caption("🌐 No filters applied - searched across all content")
    else:
        st.info("👋 Ask your first question to get started! Use the filters in the sidebar to narrow down your search.")

def render_answer_display(answer_data: Dict[str, Any], question: str):
    """Render answer with sources and metadata"""
    
    # Main answer
    st.markdown("**📝 Answer:**")
    st.markdown(answer_data['answer'])
    
    # Confidence and metadata
    col1, col2, col3 = st.columns(3)
    
    with col1:
        confidence = answer_data.get('confidence', 0)
        st.metric("Confidence", f"{confidence*100:.1f}%")
    
    with col2:
        docs_found = answer_data['metadata'].get('documents_found', 0)
        st.metric("Documents", docs_found)
    
    with col3:
        tables_found = answer_data['metadata'].get('tables_found', 0)
        st.metric("Tables", tables_found)
    
    # Sources
    sources = answer_data.get('sources', {})
    
    if sources.get('documents') or sources.get('tables'):
        st.markdown("**📚 Sources:**")
        
        # Document sources
        if sources.get('documents'):
            with st.expander(f"📄 Document Sources ({len(sources['documents'])})"):
                for i, doc in enumerate(sources['documents']):
                    st.markdown(f"""
                    **{i+1}. {doc['file_name']}**
                    - Level: {doc['level']}, Paper: {doc['paper']}
                    - Chapter: {doc.get('chapter', 'N/A')}
                    - Similarity: {doc['similarity']:.2f}
                    - Snippet: {doc['snippet']}
                    """)
        
        # Table sources
        if sources.get('tables'):
            with st.expander(f"📊 Table Sources ({len(sources['tables'])})"):
                for i, table in enumerate(sources['tables']):
                    st.markdown(f"""
                    **{i+1}. {table['file_name']} (Page {table['page_number']})**
                    - Level: {table['level']}, Paper: {table['paper']}
                    - Dimensions: {table['rows']} rows × {table['cols']} columns
                    - Similarity: {table['similarity']:.2f}
                    - Context: {table.get('context', 'N/A')}
                    """)
    
    # Suggestions
    if answer_data.get('suggestions'):
        st.markdown("**💡 Related Questions:**")
        for i, suggestion in enumerate(answer_data['suggestions']):
            # Create a safer key using string slicing instead of hash
            safe_suggestion_key = suggestion.replace(" ", "_").replace("?", "").replace(".", "")[:50]
            safe_question_key = question.replace(" ", "_").replace("?", "").replace(".", "")[:30]
            button_key = f"suggest_{i}_{safe_suggestion_key}_{safe_question_key}"
            
            if st.button(suggestion, key=button_key):
                st.session_state.suggested_question = suggestion
                st.rerun()

def render_file_management():
    """Render file management interface"""
    st.header("📊 File Management")
    st.markdown("Manage uploaded files and monitor processing status")
    
    try:
        # Get file metadata from database
        all_files = st.session_state.vector_db.get_file_metadata()
        
        if not all_files:
            st.info("No files uploaded yet. Go to the Upload Documents page to get started.")
            return
        
        # File statistics
        st.subheader("📈 Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Files", len(all_files))

        with col2:
            completed_files = len([f for f in all_files if f.get('processing_status') == 'completed'])
            st.metric("Processed", completed_files)

        with col3:
            pending_files = len([f for f in all_files if f.get('processing_status') == 'pending'])
            st.metric("Pending", pending_files)

        with col4:
            total_pages = sum(f.get('total_pages', 0) for f in all_files)
            st.metric("Total Pages", total_pages)

        # Recent batch uploads (files uploaded within 5 minutes of each other)
        if all_files:
            st.subheader("📦 Recent Batch Uploads")

            # Group files by upload time (within 5 minute windows)
            sorted_files = sorted(all_files, key=lambda x: x.get('upload_date', ''), reverse=True)
            batches = group_files_by_upload_time(sorted_files, time_window_minutes=5)

            for i, batch in enumerate(batches[:3]):  # Show last 3 batches
                if len(batch) > 1:  # Only show actual batches
                    with st.expander(f"Batch {i+1}: {len(batch)} files uploaded {format_time_ago(batch[0].get('upload_date'))}", expanded=(i==0)):
                        batch_col1, batch_col2 = st.columns([2, 1])

                        with batch_col1:
                            st.markdown("**Files in this batch:**")
                            for file_data in batch:
                                status_icon = "✅" if file_data.get('processing_status') == 'completed' else "⏳" if file_data.get('processing_status') == 'pending' else "❌"
                                st.write(f"{status_icon} {file_data.get('file_name', 'Unknown')}")

                        with batch_col2:
                            completed_in_batch = len([f for f in batch if f.get('processing_status') == 'completed'])
                            total_pages_batch = sum(f.get('total_pages', 0) for f in batch)
                            st.metric("Completed", f"{completed_in_batch}/{len(batch)}")
                            st.metric("Total Pages", total_pages_batch)
        
        # File list
        st.subheader("📋 File List")
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            level_filter = st.selectbox("Filter by Level", ["All"] + CA_LEVELS, key="mgmt_level_filter")
        
        with col2:
            status_filter = st.selectbox("Filter by Status", ["All", "completed", "pending", "error"], key="mgmt_status_filter")
        
        with col3:
            search_term = st.text_input("Search files", placeholder="Enter file name...", key="mgmt_search")
        
        # Apply filters
        filtered_files = all_files
        
        if level_filter != "All":
            filtered_files = [f for f in filtered_files if f.get('level') == level_filter]
        
        if status_filter != "All":
            filtered_files = [f for f in filtered_files if f.get('processing_status') == status_filter]
        
        if search_term:
            filtered_files = [f for f in filtered_files if search_term.lower() in f.get('file_name', '').lower()]
        
        # Display files
        for file_data in filtered_files:
            with st.expander(f"📄 {file_data.get('file_name', 'Unknown')} - {file_data.get('processing_status', 'Unknown').title()}"):
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    **File Details:**
                    - **Level:** {file_data.get('level', 'N/A')}
                    - **Paper:** {file_data.get('paper', 'N/A')}
                    - **Module:** {file_data.get('module', 'N/A') or 'N/A'}
                    - **Chapter:** {file_data.get('chapter', 'N/A') or 'N/A'}
                    """)
                
                with col2:
                    st.markdown(f"""
                    **Processing Info:**
                    - **Status:** {file_data.get('processing_status', 'Unknown')}
                    - **Total Pages:** {file_data.get('total_pages', 0)}
                    - **Upload Date:** {file_data.get('upload_date', 'Unknown')}
                    - **File ID:** {file_data.get('file_id', 'Unknown')[:20]}...
                    """)
                
                # Action buttons
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("🔄 Reprocess", key=f"reprocess_{file_data.get('id')}"):
                        st.info("Reprocessing functionality would be implemented here")
                
                with col2:
                    if st.button("📊 View Stats", key=f"stats_{file_data.get('id')}"):
                        show_file_statistics(file_data)
                
                with col3:
                    if st.button("🗑️ Delete", key=f"delete_{file_data.get('id')}"):
                        st.warning("Delete functionality would be implemented with confirmation")
    
    except Exception as e:
        st.error(f"Error loading file management data: {str(e)}")
        logger.error(f"File management error: {e}")

def group_files_by_upload_time(files: List[Dict], time_window_minutes: int = 5) -> List[List[Dict]]:
    """Group files by upload time within a time window"""
    if not files:
        return []

    batches = []
    current_batch = [files[0]]

    for file_data in files[1:]:
        current_time = parse_upload_date(file_data.get('upload_date'))
        last_time = parse_upload_date(current_batch[-1].get('upload_date'))

        if current_time and last_time:
            time_diff = abs((current_time - last_time).total_seconds()) / 60  # minutes
            if time_diff <= time_window_minutes:
                current_batch.append(file_data)
            else:
                if len(current_batch) > 1:  # Only add batches with multiple files
                    batches.append(current_batch)
                current_batch = [file_data]
        else:
            # If we can't parse dates, treat as separate
            if len(current_batch) > 1:
                batches.append(current_batch)
            current_batch = [file_data]

    # Add the last batch if it has multiple files
    if len(current_batch) > 1:
        batches.append(current_batch)

    return batches

def parse_upload_date(upload_date_str: str) -> Optional[datetime]:
    """Parse upload date string to datetime object"""
    if not upload_date_str:
        return None

    try:
        # Try different date formats
        formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%d %H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%S.%f'
        ]

        for fmt in formats:
            try:
                return datetime.strptime(upload_date_str, fmt)
            except ValueError:
                continue

        return None
    except Exception:
        return None

def format_time_ago(upload_date_str: str) -> str:
    """Format upload date as time ago"""
    upload_time = parse_upload_date(upload_date_str)
    if not upload_time:
        return "recently"

    now = datetime.now()
    diff = now - upload_time

    if diff.days > 0:
        return f"{diff.days} day{'s' if diff.days != 1 else ''} ago"
    elif diff.seconds >= 3600:
        hours = diff.seconds // 3600
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif diff.seconds >= 60:
        minutes = diff.seconds // 60
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    else:
        return "just now"

def show_file_statistics(file_data: Dict[str, Any]):
    """Show detailed statistics for a file"""
    st.markdown("---")
    st.markdown(f"**📊 Detailed Statistics for {file_data.get('file_name')}**")

    try:
        # This would query the database for chunk and table counts
        # For now, showing placeholder structure
        st.info("Detailed statistics would show:")
        st.markdown("""
        - Number of text chunks extracted
        - Number of tables found
        - Processing methods used
        - Embedding generation statistics
        - Error logs (if any)
        """)
    except Exception as e:
        st.error(f"Error loading statistics: {str(e)}")

def render_help_info():
    """Render help and information page"""
    st.header("ℹ️ Help & Information")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📖 How to Use", "🔧 Features", "❓ FAQ", "📞 Support"])
    
    with tab1:
        st.markdown("""
        ## 🚀 Getting Started
        
        ### 1. Upload Documents
        - Go to **Upload Documents** page
        - Select a CA syllabus PDF file
        - Fill in the metadata (Level, Paper, Module, Chapter, Unit)
        - Click **Upload and Process**
        
        ### 2. Ask Questions
        - Go to **Ask Questions** page
        - Use sidebar filters to narrow down your search
        - Type your question in natural language
        - Get comprehensive answers with sources
        
        ### 3. Manage Files
        - View all uploaded files in **File Management**
        - Monitor processing status
        - Filter and search through files
        """)
    
    with tab2:
        st.markdown("""
        ## ✨ Key Features
        
        ### 📊 Table-Aware Processing
        - Advanced table extraction from PDFs
        - Preserves financial data structure
        - Maintains numerical relationships
        
        ### 🧠 Smart Embeddings
        - Azure OpenAI embeddings
        - Context-aware processing
        - Metadata-enhanced search
        
        ### 🎯 Hierarchical Filtering
        - Filter by CA Level (Foundation/Intermediate/Final)
        - Narrow down by Paper, Module, Chapter, Unit
        - Progressive difficulty recommendations
        
        ### 📚 Comprehensive Sources
        - Document chunks with similarity scores
        - Table references with context
        - Precise citations and references
        
        ### 💡 Learning Assistance
        - Related question suggestions
        - Progressive learning paths
        - Context-aware responses
        """)
    
    with tab3:
        st.markdown("""
        ## ❓ Frequently Asked Questions
        
        ### Q: What file formats are supported?
        A: Currently only PDF files are supported.
        
        ### Q: How long does processing take?
        A: Processing time depends on file size and complexity. Typically 1-5 minutes per file.
        
        ### Q: Can I upload scanned PDFs?
        A: Yes, the system includes OCR capabilities for scanned documents.
        
        ### Q: How accurate are the answers?
        A: Answers include confidence scores. Higher confidence indicates better matches.
        
        ### Q: Can I search within specific chapters?
        A: Yes, use the sidebar filters to narrow down to specific modules, chapters, or units.
        
        ### Q: What if my question doesn't get good results?
        A: Try rephrasing your question, use more specific terms, or check if relevant documents are uploaded.
        """)
    
    with tab4:
        st.markdown("""
        ## 📞 Support & Technical Info
        
        ### 🏗️ Architecture
        - **Frontend**: Streamlit web application
        - **Backend**: Python with FastAPI
        - **Database**: PostgreSQL with pgvector
        - **Storage**: Appwrite for file management
        - **AI**: Azure OpenAI for embeddings and LLM
        
        ### 🔧 Processing Pipeline
        1. PDF upload and metadata tagging
        2. Multi-method content extraction (PyMuPDF, pdfplumber, camelot, tabula)
        3. Table-aware text chunking
        4. Embedding generation with Azure OpenAI
        5. Vector storage in pgvector database
        6. RAG-based question answering
        
        ### 📊 Supported Content Types
        - Text content from PDFs
        - Financial tables and schedules
        - Scanned documents (with OCR)
        - Multi-column layouts
        - Charts and diagrams (text extraction)
        
        ### 🛡️ Data Security
        - Files stored securely in Appwrite
        - Metadata organized hierarchically
        - Processing logs maintained
        - No data sharing with external parties
        """)

def main():
    """Main application entry point"""
    st.set_page_config(
        page_title="CA RAG Assistant",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize components
    if not initialize_components():
        st.stop()
    
    # Render sidebar and get current page
    current_page = render_sidebar()
    
    # Handle suggested question
    if 'suggested_question' in st.session_state:
        # Auto-fill the question and switch to questions page
        if current_page == "📚 Ask Questions":
            # The question will be auto-filled in the interface
            pass
        del st.session_state.suggested_question
    
    # Route to appropriate page
    if current_page == "📚 Ask Questions":
        render_question_interface()
    elif current_page == "📄 Upload Documents":
        render_file_upload()
    elif current_page == "📊 File Management":
        render_file_management()
    elif current_page == "ℹ️ Help & Info":
        render_help_info()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "🎓 CA RAG Assistant - Empowering Chartered Accountancy Students with AI"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
