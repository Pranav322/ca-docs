#!/bin/bash
# Run batch ingestion with dashboard monitoring
# Usage: ./run_ingest_with_dashboard.sh

echo "🚀 Starting CA Batch Ingestion with Dashboard"
echo "============================================="

# Start the dashboard in background
echo "📊 Starting Dashboard on port 8080..."
nohup uv run python dashboard.py --port 8080 > dashboard.log 2>&1 &
DASHBOARD_PID=$!
echo "   Dashboard PID: $DASHBOARD_PID"
echo "   Access at: http://$(hostname -I | awk '{print $1}'):8080"

# Wait a moment for dashboard to start
sleep 2

# Start batch ingestion with 16 workers (optimized for 64GB VM)
echo ""
echo "📥 Starting Batch Ingestion with 16 workers..."
echo "   Log file: batch_ingest.log"
echo ""

uv run python batch_ingest.py --ca-folder ca --workers 16 2>&1 | tee -a batch_ingest.log

# When done, stop dashboard
echo ""
echo "✅ Ingestion complete! Stopping dashboard..."
kill $DASHBOARD_PID 2>/dev/null

echo "Done!"
