#!/bin/bash

# EEG Stress Detection System - Development Startup Script
# This script starts both the Python backend and Next.js frontend for development

set -e

echo "🧠 Starting EEG Stress Detection System - Development Mode"
echo "=========================================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.9+ and try again."
    exit 1
fi

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18+ and try again."
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "master_eeg_analyzer.py" ] || [ ! -f "package.json" ]; then
    echo "❌ Please run this script from the project root directory."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🐍 Activating Python virtual environment..."
source venv/bin/activate

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements-api.txt

# Install Node.js dependencies
echo "📦 Installing Node.js dependencies..."
npm install

# Create environment file if it doesn't exist
if [ ! -f ".env.local" ]; then
    echo "⚙️ Creating development environment file..."
    cp env.example .env.local
    echo "✅ Created .env.local - you may want to customize the settings"
fi

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

echo ""
echo "🚀 Starting services..."
echo ""

# Start Python backend in background
echo "🐍 Starting Python EEG API server..."
python eeg_api_server.py --port 8001 --mode realtime &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Start Next.js frontend in background
echo "🌐 Starting Next.js frontend..."
npm run dev &
FRONTEND_PID=$!

echo ""
echo "✅ Services started successfully!"
echo ""
echo "📡 Available services:"
echo "  • Frontend: http://localhost:3000"
echo "  • Backend API: http://localhost:8001"
echo "  • API Docs: http://localhost:8001/docs"
echo "  • WebSocket: ws://localhost:8001/ws"
echo "  • OSC Port: 8000 (for Mind Monitor)"
echo ""
echo "🎧 To connect your Muse headband:"
echo "  1. Install Mind Monitor app on your phone"
echo "  2. Set IP to your computer's IP address"
echo "  3. Set Port to 8000"
echo "  4. Enable EEG streaming"
echo ""
echo "🛑 Press Ctrl+C to stop all services"
echo ""

# Wait for background processes
wait $BACKEND_PID $FRONTEND_PID
