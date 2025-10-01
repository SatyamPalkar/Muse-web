# 🚀 EEG Stress Detection System - Deployment Guide

This guide covers how to connect your EEG stress detection model to the frontend and deploy the complete system.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Development Setup](#development-setup)
- [Production Deployment](#production-deployment)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## 🚀 Quick Start

### Option 1: Development Mode (Recommended for Testing)

```bash
# Clone and navigate to your project
cd /path/to/your/project

# Run the development startup script
./scripts/start-dev.sh
```

This will start both the Python backend and Next.js frontend automatically.

### Option 2: Docker Development

```bash
# Start with Docker Compose
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

### Option 3: Manual Setup

```bash
# Terminal 1: Start Python backend
python eeg_api_server.py --port 8001 --mode realtime

# Terminal 2: Start Next.js frontend
npm run dev
```

## 🛠️ Development Setup

### Prerequisites

- **Python 3.9+** with pip
- **Node.js 18+** with npm
- **Muse headband** (optional, for real EEG data)
- **Mind Monitor app** (for connecting Muse to the system)

### Step 1: Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements-api.txt

# Install Node.js dependencies
npm install
```

### Step 2: Configure Environment

```bash
# Copy environment template
cp env.example .env.local

# Edit configuration (optional)
nano .env.local
```

### Step 3: Start Services

```bash
# Use the automated script
./scripts/start-dev.sh

# Or start manually:
# Terminal 1:
python eeg_api_server.py --port 8001 --mode realtime

# Terminal 2:
npm run dev
```

### Step 4: Connect Your Muse Headband

1. **Install Mind Monitor** on your phone
2. **Connect to same WiFi** as your computer
3. **Configure Mind Monitor:**
   - IP: Your computer's IP address
   - Port: 8000
   - Enable EEG streaming
4. **Start streaming** from Mind Monitor

### Step 5: Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8001
- **API Documentation**: http://localhost:8001/docs
- **WebSocket**: ws://localhost:8001/ws

## 🏭 Production Deployment

### Docker Deployment (Recommended)

```bash
# Build and start production services
./scripts/deploy.sh production up

# Check service status
./scripts/deploy.sh production status

# View logs
./scripts/deploy.sh production logs

# Check health
./scripts/deploy.sh production health
```

### Manual Production Setup

```bash
# 1. Build frontend
npm run build

# 2. Start Python backend
python eeg_api_server.py --port 8001 --mode realtime --host 0.0.0.0

# 3. Start Next.js production server
npm start
```

## 🐳 Docker Deployment

### Development with Docker

```bash
# Start development environment
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Stop services
docker-compose down
```

### Production with Docker

```bash
# Start production environment
docker-compose --profile production up -d

# Stop services
docker-compose down
```

### Custom Docker Build

```bash
# Build custom image
docker build -t eeg-system .

# Run custom container
docker run -p 3000:3000 -p 8001:8001 -p 8000:8000 eeg-system
```

## ☁️ Cloud Deployment

### AWS Deployment

1. **Create EC2 Instance:**
   ```bash
   # Use Ubuntu 20.04 LTS
   # Instance type: t3.medium or larger
   # Security groups: Allow ports 22, 80, 443, 3000, 8001, 8000
   ```

2. **Install Docker:**
   ```bash
   sudo apt update
   sudo apt install docker.io docker-compose
   sudo usermod -aG docker $USER
   ```

3. **Deploy Application:**
   ```bash
   git clone your-repo
   cd your-repo
   ./scripts/deploy.sh production up
   ```

### Heroku Deployment

1. **Create Heroku Apps:**
   ```bash
   # Frontend app
   heroku create your-eeg-frontend
   
   # Backend app
   heroku create your-eeg-backend
   ```

2. **Deploy Backend:**
   ```bash
   cd backend
   heroku git:remote -a your-eeg-backend
   git push heroku main
   ```

3. **Deploy Frontend:**
   ```bash
   cd frontend
   heroku git:remote -a your-eeg-frontend
   git push heroku main
   ```

### DigitalOcean Deployment

1. **Create Droplet:**
   - Ubuntu 20.04 LTS
   - 2GB RAM minimum
   - Enable Docker

2. **Deploy with Docker:**
   ```bash
   # SSH into droplet
   ssh root@your-droplet-ip
   
   # Clone and deploy
   git clone your-repo
   cd your-repo
   ./scripts/deploy.sh production up
   ```

## ⚙️ Configuration

### Environment Variables

Create `.env.local` (development) or `.env` (production):

```bash
# Frontend Configuration
NODE_ENV=production
EEG_API_URL=http://localhost:8001
USE_MOCK_EEG=false

# Backend Configuration
EEG_API_HOST=0.0.0.0
EEG_API_PORT=8001
EEG_MODE=realtime
OSC_PORT=8000

# Security
JWT_SECRET=your-secret-key
API_KEY_SECRET=your-api-key
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/realtime` | GET | Get current stress analysis |
| `/api/analyze` | POST | Analyze EEG sample |
| `/api/session/start` | POST | Start session analysis |
| `/api/session/stop` | POST | Stop session analysis |
| `/api/status` | GET | Get system status |
| `/ws` | WebSocket | Real-time data streaming |

### WebSocket Events

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8001/ws');

// Listen for real-time analysis
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'realtime_analysis') {
    console.log('Stress Level:', data.data.stress_level);
    console.log('Confidence:', data.data.confidence);
  }
};
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Backend Not Starting

```bash
# Check Python dependencies
pip install -r requirements-api.txt

# Check if port is available
netstat -tulpn | grep :8001

# Run with debug mode
python eeg_api_server.py --port 8001 --mode realtime --debug
```

#### 2. Frontend Not Connecting to Backend

```bash
# Check environment variables
cat .env.local

# Verify backend is running
curl http://localhost:8001/health

# Check CORS settings in backend
```

#### 3. Muse Headband Not Connecting

```bash
# Check OSC port
netstat -tulpn | grep :8000

# Enable debug mode
python eeg_api_server.py --debug

# Check Mind Monitor settings
# - IP should match your computer's IP
# - Port should be 8000
# - EEG streaming should be enabled
```

#### 4. Docker Issues

```bash
# Check Docker status
docker-compose ps

# View logs
docker-compose logs

# Rebuild images
docker-compose build --no-cache

# Clean up
docker-compose down -v
docker system prune
```

### Performance Optimization

#### 1. Reduce Latency

```bash
# Use realtime mode for lowest latency
python eeg_api_server.py --mode realtime

# Optimize WebSocket updates
# Reduce update frequency in eeg_api_server.py
```

#### 2. Scale for Multiple Users

```bash
# Use Redis for session storage
docker-compose --profile production up -d

# Configure load balancer
# Use nginx for reverse proxy
```

#### 3. Memory Optimization

```bash
# Limit buffer sizes in master_eeg_analyzer.py
# Use smaller model files
# Implement data cleanup
```

### Monitoring and Logging

#### 1. Health Checks

```bash
# Check all services
./scripts/deploy.sh production health

# Individual service checks
curl http://localhost:3000/api/health
curl http://localhost:8001/health
```

#### 2. Log Monitoring

```bash
# View real-time logs
docker-compose logs -f

# Check specific service logs
docker-compose logs eeg-api
docker-compose logs frontend
```

#### 3. Performance Monitoring

```bash
# Monitor system resources
htop

# Check Docker resource usage
docker stats

# Monitor network connections
netstat -tulpn
```

## 📞 Support

### Getting Help

1. **Check logs** for error messages
2. **Verify configuration** in environment files
3. **Test individual components** (backend, frontend, WebSocket)
4. **Check network connectivity** between services

### Common Solutions

- **Restart services** if they become unresponsive
- **Check port availability** if services fail to start
- **Verify dependencies** are properly installed
- **Clear browser cache** if frontend issues persist

### Advanced Debugging

```bash
# Enable verbose logging
export DEBUG=true
python eeg_api_server.py --debug

# Use development mode with hot reload
npm run dev

# Monitor WebSocket connections
# Check browser developer tools Network tab
```

---

## 🎉 Success!

Once deployed, your EEG stress detection system will be accessible at:

- **Frontend Dashboard**: http://your-domain:3000
- **API Documentation**: http://your-domain:8001/docs
- **Real-time WebSocket**: ws://your-domain:8001/ws

Connect your Muse headband via Mind Monitor and start monitoring stress levels in real-time! 🧠✨
