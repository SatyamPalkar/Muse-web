# 🧠 EEG Stress Detection System

A comprehensive real-time EEG stress detection system with advanced machine learning models, interactive dashboard, and seamless Muse headband integration.

## ✨ Features

- **Real-time EEG Analysis**: Live stress detection using XGBoost and BiLSTM neural networks
- **Advanced ML Models**: 18+ research-backed EEG features with bias-corrected algorithms
- **Interactive Dashboard**: Beautiful, responsive UI with real-time charts and metrics
- **Session Analysis**: Multi-window temporal analysis (30/60/90/120 seconds)
- **WebSocket Integration**: Real-time data streaming with automatic reconnection
- **Muse Integration**: Direct OSC connection via Mind Monitor app
- **Docker Deployment**: Production-ready containerized deployment
- **API Documentation**: Complete REST API with Swagger documentation

## 🚀 Quick Start

### Option 1: Development Mode (Recommended)

```bash
# Clone and navigate to your project
cd /path/to/your/project

# Run the automated development script
./scripts/start-dev.sh
```

This will automatically:
- Install all dependencies
- Start the Python backend API server
- Start the Next.js frontend
- Configure environment settings

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

## 📡 Available Services

Once started, you'll have access to:

- **Frontend Dashboard**: http://localhost:3000
- **Backend API**: http://localhost:8001
- **API Documentation**: http://localhost:8001/docs
- **WebSocket**: ws://localhost:8001/ws
- **OSC Port**: 8000 (for Mind Monitor)

## 🎧 Connecting Your Muse Headband

1. **Install Mind Monitor** on your phone
2. **Connect to same WiFi** as your computer
3. **Configure Mind Monitor:**
   - IP: Your computer's IP address
   - Port: 8000
   - Enable EEG streaming
4. **Start streaming** from Mind Monitor
5. **Watch real-time data** in the dashboard!

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Muse Headband │────│  Mind Monitor   │────│   Python API    │
│                 │    │      App        │    │   (FastAPI)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        │ WebSocket
                                                        │ REST API
                                                        │
                       ┌─────────────────┐    ┌─────────────────┐
                       │  Next.js        │────│   Dashboard     │
                       │  Frontend       │    │   (React)       │
                       └─────────────────┘    └─────────────────┘
```

## 🛠️ Technology Stack

### Backend
- **Python 3.9+** with FastAPI
- **XGBoost** for stress classification
- **PyTorch** with BiLSTM neural networks
- **OSC** for real-time Muse communication
- **WebSocket** for live data streaming

### Frontend
- **Next.js 14** with React 18
- **TypeScript** for type safety
- **Tailwind CSS** for styling
- **Recharts** for data visualization
- **Radix UI** for components

### Deployment
- **Docker** for containerization
- **Docker Compose** for orchestration
- **Nginx** for reverse proxy
- **Multi-stage builds** for optimization

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/realtime` | GET | Get current stress analysis |
| `/api/analyze` | POST | Analyze EEG sample |
| `/api/session/start` | POST | Start session analysis |
| `/api/session/stop` | POST | Stop session analysis |
| `/api/status` | GET | Get system status |
| `/ws` | WebSocket | Real-time data streaming |

## 🐳 Deployment

### Development
```bash
# Quick start
./scripts/start-dev.sh

# Docker development
npm run docker:dev
```

### Production
```bash
# Deploy with Docker
./scripts/deploy.sh production up

# Check status
./scripts/deploy.sh production status

# View logs
./scripts/deploy.sh production logs
```

### Cloud Deployment

See [DEPLOYMENT.md](./DEPLOYMENT.md) for detailed cloud deployment instructions including:
- AWS EC2 deployment
- Heroku deployment
- DigitalOcean deployment
- Kubernetes configuration

## ⚙️ Configuration

### Environment Variables

Create `.env.local` (development) or `.env` (production):

```bash
# Frontend
NODE_ENV=development
EEG_API_URL=http://localhost:8001
USE_MOCK_EEG=false

# Backend
EEG_API_HOST=0.0.0.0
EEG_API_PORT=8001
OSC_PORT=8000
```

See [env.example](./env.example) for complete configuration options.

## 🧪 Testing

### With Mock Data
```bash
# Set environment variable
export USE_MOCK_EEG=true

# Start development
npm run dev
```

### With Real Muse Data
```bash
# Start backend
python eeg_api_server.py --mode realtime

# Connect Mind Monitor with your Muse headband
# Configure IP and port 8000
```

## 📈 Performance

- **Real-time latency**: < 100ms
- **WebSocket updates**: Every 2 seconds
- **Session analysis**: Multi-threaded processing
- **Memory usage**: Optimized with rolling buffers
- **Concurrent users**: Supports multiple WebSocket connections

## 🔧 Troubleshooting

### Common Issues

1. **Backend not starting**
   ```bash
   # Check dependencies
   pip install -r requirements-api.txt
   
   # Check port availability
   netstat -tulpn | grep :8001
   ```

2. **Frontend not connecting**
   ```bash
   # Check backend health
   curl http://localhost:8001/health
   
   # Check environment variables
   cat .env.local
   ```

3. **Muse not connecting**
   ```bash
   # Check OSC port
   netstat -tulpn | grep :8000
   
   # Enable debug mode
   python eeg_api_server.py --debug
   ```

### Getting Help

- Check the [DEPLOYMENT.md](./DEPLOYMENT.md) for detailed troubleshooting
- Review logs: `docker-compose logs -f`
- Check service status: `./scripts/deploy.sh production health`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## 🙏 Acknowledgments

- **Muse** for the innovative EEG headband technology
- **Mind Monitor** for the excellent OSC streaming app
- **Research community** for EEG stress detection algorithms
- **Open source contributors** for the amazing tools and libraries

---

## 🎉 Success!

Your EEG stress detection system is now ready! Connect your Muse headband and start monitoring stress levels in real-time. 🧠✨

For detailed deployment instructions, see [DEPLOYMENT.md](./DEPLOYMENT.md).
