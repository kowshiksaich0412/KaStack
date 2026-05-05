#!/bin/bash
# Setup and run script for KaStack Conversation Analysis

echo "================================"
echo "KaStack RAG System Setup"
echo "================================"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Python 3 not found. Please install Python 3.8+"
    exit 1
fi

echo "✓ Python found"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate 2>/dev/null || . venv/Scripts/activate

echo "✓ Virtual environment activated"

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt -q

echo "✓ Dependencies installed"

# Run the system
echo "================================"
echo "Starting RAG System Backend"
echo "================================"
echo ""
echo "API will be available at: http://localhost:5000"
echo "Frontend will be available at: http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop"
echo ""

cd backend
python app.py
