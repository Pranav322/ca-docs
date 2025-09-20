#!/usr/bin/env python3
"""
Simple database connection test to verify environment variables are loaded correctly
"""
import sys
from config import DATABASE_URL
from database import VectorDatabase

def test_config_loading():
    """Test that environment variables are loaded correctly"""
    print("=" * 60)
    print("🔍 TESTING CONFIGURATION LOADING")
    print("=" * 60)
    
    print(f"DATABASE_URL: {DATABASE_URL[:50]}...")
    
    # Check if we're using default values (indicating env vars not loaded)
    if DATABASE_URL.startswith("postgresql://user:password@localhost"):
        print("❌ ERROR: Using default DATABASE_URL - environment variables not loaded!")
        return False
    else:
        print("✅ SUCCESS: Environment variables loaded correctly!")
        print("✅ Using DATABASE_URL connection method (recommended)")
        return True

def test_database_connection():
    """Test actual database connection"""
    print("\n" + "=" * 60)
    print("🔗 TESTING DATABASE CONNECTION")
    print("=" * 60)
    
    try:
        # Create VectorDatabase instance
        print("Initializing VectorDatabase...")
        db = VectorDatabase()
        print("✅ VectorDatabase initialized successfully!")
        
        # Try to get a connection
        print("Testing database connection...")
        conn = db.get_connection()
        
        if conn:
            print("✅ Database connection successful!")
            
            # Test a simple query
            cur = conn.cursor()
            cur.execute("SELECT 1;")
            result = cur.fetchone()
            if result:
                print("✅ Database query executed successfully")
            
            cur.close()
            conn.close()
            return True
        else:
            print("❌ Failed to get database connection")
            return False
            
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 CA-RAG Database Connection Test")
    print("=" * 60)
    
    # Test config loading first
    config_ok = test_config_loading()
    
    if config_ok:
        # Test database connection
        db_ok = test_database_connection()
        
        if db_ok:
            print("\n🎉 ALL TESTS PASSED! Database is ready.")
            sys.exit(0)
        else:
            print("\n💥 Database connection test failed!")
            sys.exit(1)
    else:
        print("\n💥 Configuration loading test failed!")
        sys.exit(1)