import logging
import json
import traceback
from database import VectorDatabase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_store_table():
    print("\n🧪 Testing store_tables_batch fix...")
    try:
        # Initialize Database
        db = VectorDatabase()
        print("✅ Database connection initialized.")
        
        # Create a dummy table payload
        dummy_table = {
            "file_id": "test_file_id_123",
            "file_name": "test_file_verification.pdf",
            "table_data": [{"col1": "val1", "col2": "val2"}],
            "table_html": "<table><tr><td>val1</td><td>val2</td></tr></table>",
            "embedding": [0.1] * 1536, # Dummy embedding
            "context_before": "Table Context Before",
            "context_after": "Table Context After",
            "page_number": 1,
            "table_index": 0,
            "level": "TestLevel",
            "paper": "TestPaper",
            "module": "TestModule",
            "chapter": "TestChapter",
            "unit": "TestUnit",
        }
        
        print("📤 Attempting to store table...")
        ids = db.store_tables_batch([dummy_table])
        
        if ids and len(ids) > 0:
            print(f"✅ SUCCESS! Stored table with ID: {ids[0]}")
            return True
        else:
            print("❌ FAILURE! No IDs returned.")
            return False
            
    except Exception as e:
        print(f"❌ TEST FAILED with Exception: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_store_table()
    if success:
        print("\n✨ Fix Verification PASSED!")
    else:
        print("\n💥 Fix Verification FAILED!")
        exit(1)
