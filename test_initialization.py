#!/usr/bin/env python3
"""
Test script to check which component is failing during initialization
"""

import sys
import os
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_component(name, import_func, timeout=30):
    """Test a component with timeout"""
    print(f"Testing {name}...")
    start_time = time.time()

    try:
        result = import_func()
        elapsed = time.time() - start_time
        print(f"✅ {name} OK ({elapsed:.2f}s)")
        return result
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ {name} failed after {elapsed:.2f}s: {e}")
        raise e

try:
    print("Testing component imports with timeouts...")

    # Test 1: VectorDatabase
    def test_db():
        from database import VectorDatabase
        return VectorDatabase()
    db = test_component("VectorDatabase", test_db)

    # Test 2: PDFProcessor
    def test_pdf():
        from pdf_processor import PDFProcessor
        return PDFProcessor()
    pdf_proc = test_component("PDFProcessor", test_pdf)

    # Test 3: TableProcessor
    def test_table():
        from table_processor import TableProcessor
        return TableProcessor()
    table_proc = test_component("TableProcessor", test_table)

    # Test 4: EmbeddingManager
    def test_embed():
        from embeddings import EmbeddingManager
        return EmbeddingManager()
    embed_mgr = test_component("EmbeddingManager", test_embed, timeout=60)

    # Test 5: RAGPipeline
    def test_rag():
        from rag_pipeline import RAGPipeline
        return RAGPipeline()
    rag_pipe = test_component("RAGPipeline", test_rag, timeout=60)

    # Test 6: AppwriteClient
    def test_appwrite():
        from appwrite_client import AppwriteClient
        return AppwriteClient()
    appwrite = test_component("AppwriteClient", test_appwrite, timeout=60)

    print("\n🎉 All components initialized successfully!")

except Exception as e:
    print(f"\n❌ Failed at: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)