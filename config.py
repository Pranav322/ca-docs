import os
import logging
import socket
from dotenv import load_dotenv

# --- DNS PATCH START ---
# Temporary workaround for hotspot DNS issues manually resolving the Neon Hostname
# Remove this when deploying to Azure VM or environments with working DNS
_original_getaddrinfo = socket.getaddrinfo

def _custom_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    if host == 'ep-shy-dust-ahhu9ol0-pooler.c-3.us-east-1.aws.neon.tech':
        host = '18.215.6.120'
    return _original_getaddrinfo(host, port, family, type, proto, flags)

socket.getaddrinfo = _custom_getaddrinfo
# --- DNS PATCH END ---

# Load environment variables from .env file FIRST
load_dotenv()

# Set up logging for configuration debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY", "")
AZURE_OPENAI_VERSION = os.getenv("AZURE_OPENAI_VERSION", "2024-02-01")
AZURE_EMBEDDINGS_DEPLOYMENT = os.getenv("AZURE_EMBEDDINGS_DEPLOYMENT", "text-embedding-ada-002")
AZURE_LLM_DEPLOYMENT = os.getenv("AZURE_LLM_DEPLOYMENT", "gpt-4")
AZURE_OPENAI_MINI_DEPLOYMENT = os.getenv("AZURE_OPENAI_MINI_DEPLOYMENT", "gpt-4o-mini")

# New Azure Deployment for Enrichment
NEW_AZURE_OPENAI_ENDPOINT = os.getenv("NEW_AZURE_OPENAI_ENDPOINT", "")
NEW_AZURE_OPENAI_KEY = os.getenv("NEW_AZURE_OPENAI_KEY", "")
NEW_AZURE_LLM_DEPLOYMENT = os.getenv("NEW_AZURE_LLM_DEPLOYMENT", "gpt-4o-mini")
NEW_AZURE_OPENAI_MINI_DEPLOYMENT = os.getenv("NEW_AZURE_OPENAI_MINI_DEPLOYMENT", "gpt-4o-mini")

# Database Configuration - Using DATABASE_URL for connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/ca_rag_db")

# Appwrite Configuration
APPWRITE_ENDPOINT = os.getenv("APPWRITE_ENDPOINT", "")
APPWRITE_PROJECT_ID = os.getenv("APPWRITE_PROJECT_ID", "")
APPWRITE_API_KEY = os.getenv("APPWRITE_API_KEY", "")
APPWRITE_BUCKET_ID = os.getenv("APPWRITE_BUCKET_ID", "ca-pdfs")

# Chunking Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TABLE_CONTEXT_SIZE = 500

# CA Course Structure
CA_LEVELS = ["Foundation", "Intermediate", "Final"]
CA_PAPERS = {
    "Foundation": ["Paper 1", "Paper 2", "Paper 3", "Paper 4"],
    "Intermediate": ["Group I Paper 1", "Group I Paper 2", "Group I Paper 3", "Group I Paper 4", 
                    "Group II Paper 5", "Group II Paper 6", "Group II Paper 7", "Group II Paper 8"],
    "Final": ["Group I Paper 1", "Group I Paper 2", "Group I Paper 3", "Group I Paper 4",
             "Group II Paper 5", "Group II Paper 6", "Group II Paper 7", "Group II Paper 8"]
}
