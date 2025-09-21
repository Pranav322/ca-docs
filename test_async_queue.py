#!/usr/bin/env python3
"""
Test script for AsyncFileQueueManager
Run this to verify the asyncio queue implementation works correctly
"""

import asyncio
import sys
import os

# Add current directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from async_queue_manager import AsyncFileQueueManager, get_queue_manager
    from database import VectorDatabase
    print("✅ Successfully imported AsyncFileQueueManager and VectorDatabase")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

async def test_queue_manager():
    """Test the async queue manager functionality"""
    print("\n🧪 Testing AsyncFileQueueManager...")

    try:
        # Test 1: Create queue manager
        print("1. Creating queue manager...")
        queue_manager = get_queue_manager(max_workers=2)
        print("✅ Queue manager created successfully")

        # Test 2: Start workers
        print("2. Starting workers...")
        await queue_manager.start_workers()
        print("✅ Workers started successfully")

        # Test 3: Test database connection
        print("3. Testing database connection...")
        db = VectorDatabase()
        # Try to get file metadata (should work even if empty)
        files = db.get_file_metadata()
        print(f"✅ Database connection successful (found {len(files)} files)")

        # Test 4: Test session creation
        print("4. Testing session creation...")
        session_id = "test_session_" + str(asyncio.get_event_loop().time())
        session_db_id = db.create_upload_session(session_id, 5)
        print(f"✅ Session created with ID: {session_db_id}")

        # Test 5: Test file addition to queue
        print("5. Testing file addition to queue...")
        file_id = "test_file_" + str(asyncio.get_event_loop().time())
        file_name = "test_document.pdf"
        metadata = {
            'level': 'Foundation',
            'paper': 'Paper 1: Accounting',
            'chapter': 'Chapter 1: Introduction'
        }

        queue_id = db.add_file_to_queue(session_id, file_id, file_name, metadata)
        print(f"✅ File added to queue with ID: {queue_id}")

        # Test 6: Test session status retrieval
        print("6. Testing session status retrieval...")
        session_status = db.get_session_status(session_id)
        if 'error' not in session_status:
            print(f"✅ Session status retrieved: {session_status['total_files']} files")
        else:
            print(f"❌ Session status error: {session_status['error']}")

        # Test 7: Stop workers
        print("7. Stopping workers...")
        await queue_manager.stop_workers()
        print("✅ Workers stopped successfully")

        print("\n🎉 All tests passed! Async queue system is ready.")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 Testing Asyncio Queue System for CA RAG Assistant")
    print("=" * 60)

    # Run async tests
    try:
        result = asyncio.run(test_queue_manager())
        if result:
            print("\n✅ All tests completed successfully!")
            print("\n📋 Next steps:")
            print("1. Run your Streamlit app: streamlit run app.py")
            print("2. Go to Upload Documents page")
            print("3. Try uploading multiple PDF files")
            print("4. Check File Management page for progress tracking")
        else:
            print("\n❌ Some tests failed. Please check the errors above.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()