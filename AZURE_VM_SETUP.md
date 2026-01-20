# Azure VM Deployment Guide for CA Batch Ingestion

## Quick Deployment Steps

### 1. SSH into your VM
```bash
ssh azureuser@<your-vm-ip>
```

### 2. Install Dependencies
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and system deps
sudo apt install -y python3 python3-pip git ghostscript poppler-utils

# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

### 3. Clone and Setup Project
```bash
git clone <your-repo-url> ca-docs-backend
cd ca-docs-backend
uv sync
```

### 4. Configure Environment
```bash
# Create .env file with your credentials
cat > .env << 'EOF'
AZURE_OPENAI_ENDPOINT=https://aiservices-orizn.openai.azure.com/
AZURE_OPENAI_KEY=<your-key>
AZURE_LLM_DEPLOYMENT=gpt-4.1
AZURE_EMBEDDINGS_DEPLOYMENT=text-embedding-3-small
DATABASE_URL=<your-neon-connection-string>
APPWRITE_ENDPOINT=https://fra.cloud.appwrite.io/v1
APPWRITE_PROJECT_ID=<your-project-id>
APPWRITE_API_KEY=<your-api-key>
EOF
```

### 5. Upload PDFs
```bash
# Option A: SCP from local
scp -r ca/ azureuser@<vm-ip>:~/ca-docs-backend/

# Option B: Git LFS (if PDFs are in repo)
git lfs pull
```

### 6. Remove DNS Patch (Important!)
Edit `config.py` and **comment out** lines 7-17 (the DNS patch):
```bash
nano config.py
# Comment out the socket.getaddrinfo override section
```

### 7. Open Firewall Port for Dashboard
```bash
# Azure CLI (run from local)
az vm open-port --port 8080 --resource-group <your-rg> --name <vm-name>
```

---

## Running the Ingestion

### Option A: Foreground (see output)
```bash
# Terminal 1: Dashboard
uv run python dashboard.py --port 8080 &

# Terminal 2: Ingestion with 16 workers
uv run python batch_ingest.py --ca-folder ca --workers 16
```

### Option B: Background with tmux (Recommended)
```bash
# Start tmux session
tmux new -s ingest

# Run both
./run_ingest_with_dashboard.sh

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t ingest
```

### Option C: Background with nohup
```bash
# Start dashboard
nohup uv run python dashboard.py --port 8080 > dashboard.log 2>&1 &

# Start ingestion
nohup uv run python batch_ingest.py --ca-folder ca --workers 16 > ingest.log 2>&1 &

# Check progress
tail -f ingest.log
```

---

## Monitoring

### Dashboard
Open in browser: `http://<vm-ip>:8080`

### Command Line
```bash
# Live logs
tail -f batch_ingest.log

# Stats via API
curl http://localhost:8080/api/stats | jq

# Failed files
curl http://localhost:8080/api/files?status=failed | jq
```

---

## Performance Tuning

| RAM | Recommended Workers |
|-----|---------------------|
| 8GB | 4-8 |
| 16GB | 8-12 |
| 32GB | 12-20 |
| 64GB | 16-24 |

Your 64GB VM can comfortably handle **16-24 workers**.

### Speed Tips
1. **LLM fallback is disabled** - Using smart regex only (10x faster)
2. **Increase workers** - More parallel PDF processing
3. **SSD storage** - Faster temp file operations

---

## Troubleshooting

### Connection Pool Exhausted
Dashboard and ingestion share connections. This is fixed in latest version.

### DNS Issues
On Azure VM, **remove the DNS patch** from `config.py` (lines 7-17).

### Rate Limiting
If you see 429 errors from Azure OpenAI, reduce workers:
```bash
uv run python batch_ingest.py --workers 8
```
