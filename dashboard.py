"""
Batch Ingestion Dashboard
A simple FastAPI dashboard to monitor batch processing progress
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import uvicorn
from datetime import datetime
import json
import os
import re
from typing import Optional
import psycopg2
from psycopg2.extras import RealDictCursor
from config import DATABASE_URL
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CA Batch Ingest Dashboard", version="1.0")


def get_db_connection():
    """Get a fresh database connection (not pooled, to avoid conflicts)"""
    try:
        return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return None


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Main dashboard page"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CA Batch Ingest Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', system-ui, sans-serif; 
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh; 
            color: #fff; 
            padding: 2rem;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { 
            font-size: 2rem; 
            margin-bottom: 2rem; 
            background: linear-gradient(90deg, #00d9ff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .stats-grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
            gap: 1.5rem; 
            margin-bottom: 2rem; 
        }
        .stat-card { 
            background: rgba(255,255,255,0.05); 
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 16px; 
            padding: 1.5rem;
            backdrop-filter: blur(10px);
        }
        .stat-card h3 { font-size: 0.9rem; color: #888; margin-bottom: 0.5rem; }
        .stat-card .value { font-size: 2rem; font-weight: 700; }
        .stat-card.success .value { color: #00ff88; }
        .stat-card.error .value { color: #ff4757; }
        .stat-card.pending .value { color: #ffa502; }
        .stat-card.processing .value { color: #00d9ff; }
        .stat-card.info .value { color: #00d9ff; }
        .progress-section { 
            background: rgba(255,255,255,0.05); 
            border-radius: 16px; 
            padding: 2rem; 
            margin-bottom: 2rem;
        }
        .progress-bar { 
            background: rgba(255,255,255,0.1); 
            border-radius: 10px; 
            height: 20px; 
            overflow: hidden; 
            margin-top: 1rem;
        }
        .progress-fill { 
            height: 100%; 
            background: linear-gradient(90deg, #00d9ff, #00ff88);
            transition: width 0.5s ease;
        }
        .log-section { 
            background: rgba(0,0,0,0.3); 
            border-radius: 16px; 
            padding: 1.5rem;
            max-height: 400px;
            overflow-y: auto;
        }
        .log-section h2 { margin-bottom: 1rem; font-size: 1.2rem; }
        .log-entry { 
            padding: 0.5rem; 
            border-bottom: 1px solid rgba(255,255,255,0.05);
            font-family: monospace;
            font-size: 0.9rem;
        }
        .log-entry.error { color: #ff4757; }
        .log-entry.success { color: #00ff88; }
        .refresh-btn {
            background: linear-gradient(90deg, #00d9ff, #00ff88);
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            color: #000;
            font-weight: 600;
            cursor: pointer;
            margin-bottom: 1rem;
        }
        .auto-refresh { color: #888; font-size: 0.9rem; margin-left: 1rem; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 CA Batch Ingest Dashboard</h1>
        
        <button class="refresh-btn" onclick="fetchStats()">🔄 Refresh Now</button>
        <span class="auto-refresh">Auto-refreshes every 10s</span>
        
        <div class="stats-grid" id="stats-grid">
            <div class="stat-card info"><h3>Total Documents</h3><div class="value" id="total-docs">-</div></div>
            <div class="stat-card info"><h3>Total Tables</h3><div class="value" id="total-tables">-</div></div>
            <div class="stat-card success"><h3>Completed Files</h3><div class="value" id="completed">-</div></div>
            <div class="stat-card processing"><h3>Processing</h3><div class="value" id="processing">-</div></div>
            <div class="stat-card pending"><h3>Pending Files</h3><div class="value" id="pending">-</div></div>
            <div class="stat-card error"><h3>Failed Files</h3><div class="value" id="failed">-</div></div>
            <div class="stat-card info"><h3>Total Files</h3><div class="value" id="total-files">-</div></div>
        </div>
        
        <div class="progress-section">
            <h2>Processing Progress</h2>
            <div id="progress-text">Loading...</div>
            <div class="progress-bar">
                <div class="progress-fill" id="progress-fill" style="width: 0%"></div>
            </div>
        </div>
        
        <div class="log-section">
            <h2>Recent Activity (from batch_ingest.log)</h2>
            <div id="log-entries">Loading...</div>
        </div>
    </div>
    
    <script>
        async function fetchStats() {
            try {
                const res = await fetch('/api/stats');
                const data = await res.json();
                
                document.getElementById('total-docs').textContent = data.document_count || 0;
                document.getElementById('total-tables').textContent = data.table_count || 0;
                
                const discovered = data.discovered_count || 0;
                document.getElementById('total-files').textContent = discovered || data.file_count || 0;
                
                const status = data.processing_status || {};
                const completed = status.completed || 0;
                const processing = status.processing || 0;
                const pending = status.pending || 0;
                const failed = status.failed || 0;
                const total = discovered || (completed + processing + pending + failed);
                
                document.getElementById('completed').textContent = completed;
                document.getElementById('processing').textContent = processing;
                document.getElementById('pending').textContent = pending;
                document.getElementById('failed').textContent = failed;
                
                const progress = total > 0 ? ((completed + processing) / total * 100) : 0;
                document.getElementById('progress-fill').style.width = progress + '%';
                document.getElementById('progress-text').textContent = 
                    `${completed} completed, ${processing} processing / ${total} total (${progress.toFixed(1)}%)`;
                    
            } catch (e) {
                console.error('Failed to fetch stats:', e);
            }
        }
        
        async function fetchLogs() {
            try {
                const res = await fetch('/api/logs?lines=50');
                const data = await res.json();
                
                const container = document.getElementById('log-entries');
                container.innerHTML = data.logs.map(log => {
                    const cls = log.includes('ERROR') ? 'error' : 
                                log.includes('Successfully') ? 'success' : '';
                    return `<div class="log-entry ${cls}">${log}</div>`;
                }).join('');
                
            } catch (e) {
                console.error('Failed to fetch logs:', e);
            }
        }
        
        // Initial fetch
        fetchStats();
        fetchLogs();
        
        // Auto-refresh
        setInterval(fetchStats, 10000);
        setInterval(fetchLogs, 5000);
    </script>
</body>
</html>
    """


@app.get("/api/stats")
async def get_stats():
    """Get current processing statistics"""
    stats = {
        "document_count": 0,
        "table_count": 0,
        "file_count": 0,
        "processing_status": {},
    }

    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor()

            # Document count
            cur.execute("SELECT COUNT(*) as count FROM documents;")
            result = cur.fetchone()
            stats["document_count"] = result["count"] if result else 0

            # Table count
            cur.execute("SELECT COUNT(*) as count FROM tables;")
            result = cur.fetchone()
            stats["table_count"] = result["count"] if result else 0

            # File count
            cur.execute("SELECT COUNT(*) as count FROM file_metadata;")
            result = cur.fetchone()
            stats["file_count"] = result["count"] if result else 0

            # Processing status breakdown
            cur.execute("""
                SELECT processing_status, COUNT(*) as count 
                FROM file_metadata 
                GROUP BY processing_status;
            """)
            stats["processing_status"] = {
                row["processing_status"]: row["count"] for row in cur.fetchall()
            }

            cur.close()
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
        finally:
            conn.close()

    # Also parse log for discovered files count
    log_file = "batch_ingest.log"
    discovered_count = 0
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            for line in f:
                if "Discovered" in line and "PDF files" in line:
                    match = re.search(r"Discovered (\d+) PDF", line)
                    if match:
                        discovered_count = int(match.group(1))

    stats["discovered_count"] = discovered_count
    return stats


@app.get("/api/logs")
async def get_logs(lines: int = 50):
    """Get recent log entries"""
    try:
        log_file = "batch_ingest.log"
        if not os.path.exists(log_file):
            return {"logs": ["No log file found. Start batch ingestion first."]}

        with open(log_file, "r") as f:
            all_lines = f.readlines()
            recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
            return {"logs": [line.strip() for line in reversed(recent_lines)]}
    except Exception as e:
        logger.error(f"Failed to read logs: {e}")
        return {"logs": [f"Error reading logs: {e}"]}


@app.get("/api/files")
async def get_files(status: Optional[str] = None):
    """Get file metadata with optional status filter"""
    files = []
    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor()
            if status:
                cur.execute(
                    "SELECT * FROM file_metadata WHERE processing_status = %s;",
                    (status,),
                )
            else:
                cur.execute("SELECT * FROM file_metadata LIMIT 100;")
            files = [dict(row) for row in cur.fetchall()]
            cur.close()
        except Exception as e:
            logger.error(f"Failed to get files: {e}")
        finally:
            conn.close()

    return {"files": files, "count": len(files)}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    args = parser.parse_args()

    print(f"🚀 Dashboard running at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
