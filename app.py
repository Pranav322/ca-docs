import streamlit as st
import os
import tempfile
import uuid
from typing import Dict, Any, List, Optional
import logging
import time

# Import our custom modules
from database import VectorDatabase
from pdf_processor import PDFProcessor
from table_processor import TableProcessor
from embeddings import EmbeddingManager
from rag_pipeline import RAGPipeline
from appwrite_client import AppwriteClient
from utils import FileUtils, ValidationUtils, ProgressTracker, ResponseFormatter
from config import CA_LEVELS, CA_PAPERS
from curriculum_ui import CurriculumSelector, render_curriculum_filter
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
        
        # Use the new curriculum filter component
        with st.sidebar:
            filters = render_curriculum_filter(prefix="sidebar_filter", title="", show_clear=True)
            
            include_tables = st.checkbox("Include Tables", value=True, key="include_tables")
    
    return page

def render_file_upload():
    """Render file upload interface"""
    st.header("📄 Upload CA Study Materials")
    st.markdown("Upload PDF files from CA syllabus with proper metadata tagging")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose PDF file",
        type=['pdf'],
        help="Upload CA syllabus PDFs (Foundation, Intermediate, or Final level)"
    )
    
    if uploaded_file is not None:
        # Display file info
        st.success(f"File selected: {uploaded_file.name} ({uploaded_file.size} bytes)")
        
        # Smart Curriculum-based Metadata Tagging
        st.subheader("🏷️ Smart Curriculum Tagging")
        st.markdown("Select the curriculum hierarchy to automatically tag your document")
        
        # Initialize curriculum selector for upload
        upload_selector = CurriculumSelector(prefix="upload")
        
        # Render the curriculum selector
        selection = upload_selector.render_complete_selector(
            title="📖 Select Document Location in Curriculum",
            show_path=True,
            columns=True
        )
        
        # Show selection status
        upload_selector.render_selection_status()
        
        # Additional metadata form
        with st.form("metadata_form"):
            # Additional metadata
            with st.expander("📋 Additional Information"):
                description = st.text_area("Description (optional)", key="upload_description")
                tags = st.text_input("Tags (comma-separated)", key="upload_tags")
            
            submitted = st.form_submit_button("🚀 Upload and Process")
            
            if submitted:
                if not upload_selector.is_complete_selection():
                    missing = upload_selector.get_missing_selections()
                    st.error(f"Please complete the curriculum selection: {', '.join(missing)}")
                else:
                    # Get the final selection from the curriculum selector
                    final_selection = upload_selector.get_current_selection()
                    
                    process_uploaded_file(uploaded_file, {
                        'level': final_selection['level'],
                        'paper': final_selection['paper'],
                        'module': final_selection['module'],
                        'chapter': final_selection['chapter'],
                        'unit': final_selection['unit'],
                        'description': description or None,
                        'tags': [tag.strip() for tag in tags.split(',') if tag.strip()] if tags else []
                    })

def process_uploaded_file(uploaded_file, metadata: Dict[str, Any]):
    """Process uploaded PDF file"""
    try:
        # Generate unique file ID
        file_id = FileUtils.generate_file_id(uploaded_file.name)
        
        # Add file name to metadata
        metadata['file_name'] = uploaded_file.name
        
        # Validate metadata
        validated_metadata = ValidationUtils.validate_metadata(metadata)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Validate PDF
        if not FileUtils.validate_pdf_file(tmp_file_path):
            st.error("Invalid PDF file. Please upload a valid PDF.")
            FileUtils.cleanup_temp_file(tmp_file_path)
            return
        
        # Initialize progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Upload to Appwrite
        status_text.text("📤 Uploading file to storage...")
        progress_bar.progress(10)
        
        appwrite_file_id = st.session_state.appwrite_client.upload_file(tmp_file_path, uploaded_file.name)
        
        # Step 2: Store metadata
        status_text.text("💾 Storing file metadata...")
        progress_bar.progress(20)
        
        st.session_state.vector_db.store_file_metadata(
            file_id=file_id,
            file_name=uploaded_file.name,
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
            file_name=uploaded_file.name,
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
                    file_name=uploaded_file.name,
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
                    file_name=uploaded_file.name,
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
        
        # Show summary
        st.success(f"""
        **File processed successfully!**
        - **File:** {uploaded_file.name}
        - **Level:** {validated_metadata['level']}
        - **Paper:** {validated_metadata['paper']}
        - **Total Pages:** {total_pages}
        - **Text Chunks:** {len(pdf_results['text_chunks'])}
        - **Tables Found:** {len(pdf_results['tables'])}
        """)
        
        # Add to session state
        st.session_state.uploaded_files.append({
            'file_id': file_id,
            'file_name': uploaded_file.name,
            'metadata': validated_metadata,
            'status': 'completed',
            'upload_time': time.time()
        })
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        logger.error(f"File processing error: {e}")
        
        # Cleanup on error
        try:
            if 'tmp_file_path' in locals():
                FileUtils.cleanup_temp_file(tmp_file_path)
        except:
            pass

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
        
        # Get filters from the new curriculum selector in sidebar
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
            if st.button(suggestion, key=f"suggest_{i}_{hash(suggestion)}_{hash(question)}"):
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
