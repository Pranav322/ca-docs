from database import VectorDatabase
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def reset_database():
    print("\n🔥 WARNING: This will DELETE ALL DATA from the database.")
    print("Tables: documents, tables, file_metadata, question_logs")
    
    confirm = input("Type 'DELETE' to confirm: ")
    if confirm != 'DELETE':
        print("Aborted.")
        return

    try:
        db = VectorDatabase()
        conn = db.get_connection()
        cur = conn.cursor()
        
        # Truncate in correct order
        cur.execute("TRUNCATE TABLE documents, tables, file_metadata, question_logs RESTART IDENTITY CASCADE;")
        
        conn.commit()
        cur.close()
        conn.close()
        print("✅ Database Wiped Clean.")
        
    except Exception as e:
        print(f"❌ Failed to reset DB: {e}")

if __name__ == "__main__":
    reset_database()
