#!/bin/bash

# EEG Stress Detection System - Production Deployment Script
# This script builds and deploys the complete system using Docker

set -e

echo "🚀 EEG Stress Detection System - Production Deployment"
echo "====================================================="

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker and try again."
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose and try again."
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "docker-compose.yml" ] || [ ! -f "Dockerfile" ]; then
    echo "❌ Please run this script from the project root directory."
    exit 1
fi

# Parse command line arguments
ENVIRONMENT=${1:-production}
ACTION=${2:-up}

echo "📋 Deployment Configuration:"
echo "  Environment: $ENVIRONMENT"
echo "  Action: $ACTION"
echo ""

# Create environment file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "⚙️ Creating production environment file..."
    cp env.example .env
    echo "⚠️  Please edit .env file with your production settings before continuing"
    echo "   Press Enter to continue after editing, or Ctrl+C to exit"
    read
fi

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Deployment interrupted"
    exit 1
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

case $ACTION in
    "build")
        echo "🔨 Building Docker images..."
        docker-compose build
        echo "✅ Build completed"
        ;;
    
    "up")
        echo "🚀 Starting services..."
        if [ "$ENVIRONMENT" = "production" ]; then
            docker-compose --profile production up -d
        else
            docker-compose up -d
        fi
        echo "✅ Services started"
        ;;
    
    "down")
        echo "🛑 Stopping services..."
        docker-compose down
        echo "✅ Services stopped"
        ;;
    
    "restart")
        echo "🔄 Restarting services..."
        docker-compose down
        sleep 2
        if [ "$ENVIRONMENT" = "production" ]; then
            docker-compose --profile production up -d
        else
            docker-compose up -d
        fi
        echo "✅ Services restarted"
        ;;
    
    "logs")
        echo "📋 Showing service logs..."
        docker-compose logs -f
        ;;
    
    "status")
        echo "📊 Service status:"
        docker-compose ps
        ;;
    
    "health")
        echo "🏥 Checking service health..."
        
        # Check if services are running
        if ! docker-compose ps | grep -q "Up"; then
            echo "❌ No services are running"
            exit 1
        fi
        
        # Check frontend health
        echo "🌐 Checking frontend health..."
        if curl -f http://localhost:3000/api/health > /dev/null 2>&1; then
            echo "✅ Frontend is healthy"
        else
            echo "❌ Frontend is not responding"
        fi
        
        # Check backend health
        echo "🐍 Checking backend health..."
        if curl -f http://localhost:8001/health > /dev/null 2>&1; then
            echo "✅ Backend is healthy"
        else
            echo "❌ Backend is not responding"
        fi
        ;;
    
    *)
        echo "❌ Unknown action: $ACTION"
        echo ""
        echo "Usage: $0 [environment] [action]"
        echo ""
        echo "Environments:"
        echo "  production  - Production deployment with all services"
        echo "  development - Development deployment"
        echo ""
        echo "Actions:"
        echo "  build   - Build Docker images"
        echo "  up      - Start services"
        echo "  down    - Stop services"
        echo "  restart - Restart services"
        echo "  logs    - Show service logs"
        echo "  status  - Show service status"
        echo "  health  - Check service health"
        echo ""
        echo "Examples:"
        echo "  $0 production up     # Start production deployment"
        echo "  $0 development up    # Start development deployment"
        echo "  $0 production health # Check production health"
        exit 1
        ;;
esac

if [ "$ACTION" = "up" ] || [ "$ACTION" = "restart" ]; then
    echo ""
    echo "✅ Deployment completed successfully!"
    echo ""
    echo "📡 Available services:"
    echo "  • Frontend: http://localhost:3000"
    echo "  • Backend API: http://localhost:8001"
    echo "  • API Docs: http://localhost:8001/docs"
    echo "  • WebSocket: ws://localhost:8001/ws"
    echo "  • OSC Port: 8000 (for Mind Monitor)"
    echo ""
    echo "🔍 To check service status: $0 $ENVIRONMENT status"
    echo "📋 To view logs: $0 $ENVIRONMENT logs"
    echo "🏥 To check health: $0 $ENVIRONMENT health"
    echo ""
fi
