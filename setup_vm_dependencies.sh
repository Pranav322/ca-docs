#!/bin/bash

echo "🚀 Setting up VM dependencies for Batch Ingestion..."

# Update package list
echo "📦 Updating package list..."
sudo apt-get update

# Install libgl1 (Required for OpenCV)
echo "📷 Installing libgl1 (for OpenCV)..."
sudo apt-get install -y libgl1

# Install default-jre (Required for Tabula Java)
echo "☕ Installing default-jre (for Tabula)..."
sudo apt-get install -y default-jre

# Install tesseract-ocr (Optional fallback)
echo "📝 Installing tesseract-ocr (Fallback)..."
sudo apt-get install -y tesseract-ocr

echo "✅ Dependencies installed!"
echo "Now run: python verify_db_fix.py"
