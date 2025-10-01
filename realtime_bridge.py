"""
Real-Time EEG Data Bridge with WebSocket Support
Provides instant real-time updates via WebSocket
"""

import time
import json
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Real-Time EEG Data Bridge", version="3.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RealTimeEEGBridge:
    def __init__(self):
        self.eeg_analyzer_port = 9000
        self.samples_collected = 0
        self.session_start_time = time.time()
        self.is_collecting = False
        self.session_duration = 120
        
        # Track real EEG data
        self.has_real_eeg_data = False
        self.last_eeg_data_time = None
        self.data_timeout = 15
        
        # Store actual brainwave data
        self.current_brainwaves = {
            'theta': 0,
            'alpha': 0,
            'beta': 0,
            'gamma': 0,
            'delta': 0
        }
        self.brainwave_history = []
        
        # WebSocket connections
        self.active_connections: list[WebSocket] = []
        
    async def _stream_data_to_websockets(self):
        """Continuously stream data to all connected WebSocket clients."""
        while True:
            try:
                if self.active_connections:
                    # Get current analysis
                    analysis = self.get_analysis()
                    
                    # Send to all connected clients
                    message = {
                        "type": "realtime_analysis",
                        "data": analysis
                    }
                    
                    # Send to all active connections
                    disconnected = []
                    for connection in self.active_connections:
                        try:
                            await connection.send_text(json.dumps(message))
                        except:
                            disconnected.append(connection)
                    
                    # Remove disconnected clients
                    for connection in disconnected:
                        if connection in self.active_connections:
                            self.active_connections.remove(connection)
                            print(f"Removed disconnected WebSocket client")
                
                # Stream at 10Hz (100ms intervals) for true real-time
                await asyncio.sleep(0.1)
                
            except Exception as e:
                print(f"Error in background streaming: {e}")
                await asyncio.sleep(1)
        
    def receive_real_eeg_data(self, theta, alpha, beta, gamma, delta):
        """Receive real EEG data from Mind Monitor via your analyzer."""
        self.current_brainwaves = {
            'theta': theta,
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
            'delta': delta
        }
        self.has_real_eeg_data = True
        self.last_eeg_data_time = time.time()
        
        # Store in history for charts
        brainwave_entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "theta": theta,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "delta": delta
        }
        self.brainwave_history.append(brainwave_entry)
        
        # Keep only last 50 entries for performance
        if len(self.brainwave_history) > 50:
            self.brainwave_history = self.brainwave_history[-50:]
            
        print(f"🧠 Real EEG data received: θ={theta:.1f}, α={alpha:.1f}, β={beta:.1f}, γ={gamma:.1f}, δ={delta:.1f}")
        
    def is_eeg_data_fresh(self):
        """Check if we have fresh EEG data from Mind Monitor."""
        if not self.has_real_eeg_data or not self.last_eeg_data_time:
            return False
        return (time.time() - self.last_eeg_data_time) < self.data_timeout
        
    def get_analysis(self):
        """Get analysis - either real data or 'No Data' message."""
        current_time = time.time()
        
        # Check if we have fresh real EEG data
        if not self.is_eeg_data_fresh():
            # NO REAL DATA - Show proper "No Data" state
            return {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "stress_level": "No Data",
                "confidence": 0.0,
                "overall_stress": 0.0,
                "arousal_level": 0.0,
                "temporal_trend": "No Data",
                "recommendations": [
                    "🔌 Connect your Muse headband to Mind Monitor",
                    "📱 Start EEG streaming in Mind Monitor app",
                    "⚙️ Ensure Mind Monitor sends data to port 9000",
                    "🧠 Place headband on your head properly"
                ],
                "stress_indicators": {
                    "beta_alpha_ratio": 0.0,
                    "stress_index": 0.0,
                    "rel_theta": 0.0,
                    "rel_alpha": 0.0,
                    "rel_beta": 0.0,
                    "rel_gamma": 0.0,
                },
                "sample_count": 0,
                "brainwaves": {
                    'theta': 0,
                    'alpha': 0,
                    'beta': 0,
                    'gamma': 0,
                    'delta': 0
                },
                "data_status": "no_sensor",
                "message": "Waiting for EEG data from Mind Monitor..."
            }
        
        # WE HAVE REAL DATA - Process it
        if self.is_collecting:
            # Increment samples only when we have real data
            if current_time - self.session_start_time >= 1.0:
                self.samples_collected += 1
                self.session_start_time = current_time
        
        # Calculate stress analysis from real brainwave data
        beta_alpha_ratio = self.current_brainwaves['beta'] / max(1.0, self.current_brainwaves['alpha'])
        
        # Determine stress level based on real ratios
        if beta_alpha_ratio > 1.2:
            stress_level = "Moderate Stress"
            overall_stress = 0.7
        elif beta_alpha_ratio > 0.9:
            stress_level = "Light Stress"
            overall_stress = 0.5
        elif beta_alpha_ratio < 0.6:
            stress_level = "Relaxed"
            overall_stress = 0.2
        else:
            stress_level = "Neutral"
            overall_stress = 0.4
        
        # Calculate confidence based on data freshness
        time_since_data = time.time() - self.last_eeg_data_time
        confidence = max(0.5, 1.0 - (time_since_data / self.data_timeout))
        
        return {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "stress_level": stress_level,
            "confidence": confidence,
            "overall_stress": overall_stress,
            "arousal_level": overall_stress,
            "temporal_trend": "Real Data",
            "recommendations": [
                "✅ Real EEG data detected",
                "🧠 Brain activity being monitored",
                "📊 Live stress analysis active"
            ],
            "stress_indicators": {
                "beta_alpha_ratio": beta_alpha_ratio,
                "stress_index": overall_stress - 0.5,
                "rel_theta": self.current_brainwaves['theta'] / 100.0,
                "rel_alpha": self.current_brainwaves['alpha'] / 100.0,
                "rel_beta": self.current_brainwaves['beta'] / 100.0,
                "rel_gamma": self.current_brainwaves['gamma'] / 100.0,
            },
            "sample_count": self.samples_collected,
            "brainwaves": self.current_brainwaves.copy(),
            "data_status": "real_data",
            "message": f"Live EEG data - {stress_level}"
        }

# Initialize the bridge
realtime_bridge = RealTimeEEGBridge()

@app.on_event("startup")
async def startup_event():
    """Start background streaming when the server starts."""
    asyncio.create_task(realtime_bridge._stream_data_to_websockets())

@app.get("/")
async def root():
    return {
        "message": "Real-Time EEG Data Bridge",
        "version": "3.0.0",
        "status": "Ready - Streaming real-time EEG data via WebSocket",
        "websocket_connections": len(realtime_bridge.active_connections)
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "has_real_eeg_data": realtime_bridge.is_eeg_data_fresh(),
        "eeg_analyzer_port": realtime_bridge.eeg_analyzer_port,
        "samples_collected": realtime_bridge.samples_collected,
        "data_status": "real_data" if realtime_bridge.is_eeg_data_fresh() else "no_sensor",
        "websocket_connections": len(realtime_bridge.active_connections)
    }

@app.get("/api/realtime")
async def get_realtime():
    """Get current real-time EEG analysis."""
    try:
        analysis = realtime_bridge.get_analysis()
        return {"success": True, "data": analysis}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/status")
async def get_status():
    """Get current system status."""
    elapsed = time.time() - realtime_bridge.session_start_time if realtime_bridge.is_collecting else 0
    remaining = max(0, realtime_bridge.session_duration - elapsed)
    
    return {
        "mode": "realtime",
        "is_collecting": realtime_bridge.is_collecting and realtime_bridge.is_eeg_data_fresh(),
        "samples_collected": realtime_bridge.samples_collected if realtime_bridge.is_eeg_data_fresh() else 0,
        "realtime_buffer_size": 0 if not realtime_bridge.is_eeg_data_fresh() else min(30, realtime_bridge.samples_collected),
        "session_duration": elapsed,
        "time_remaining": remaining,
        "osc_port": 9000,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "has_real_eeg_data": realtime_bridge.is_eeg_data_fresh(),
        "data_status": "real_data" if realtime_bridge.is_eeg_data_fresh() else "no_sensor",
        "websocket_connections": len(realtime_bridge.active_connections)
    }

@app.post("/api/session/start")
async def start_session(config: dict = {"duration": 120}):
    """Start a new session (only works with real data)."""
    realtime_bridge.is_collecting = True
    realtime_bridge.session_duration = config.get("duration", 120)
    realtime_bridge.session_start_time = time.time()
    realtime_bridge.samples_collected = 0
    
    return {
        "success": True,
        "message": "Session started" if realtime_bridge.is_eeg_data_fresh() else "Session started (waiting for EEG data)",
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "duration": realtime_bridge.session_duration,
        "samples_collected": realtime_bridge.samples_collected,
        "has_real_data": realtime_bridge.is_eeg_data_fresh()
    }

@app.post("/api/session/stop")
async def stop_session():
    """Stop current session."""
    realtime_bridge.is_collecting = False
    session_duration = time.time() - realtime_bridge.session_start_time
    
    return {
        "success": True,
        "message": "Session stopped",
        "total_samples": realtime_bridge.samples_collected,
        "session_duration": session_duration,
        "had_real_data": realtime_bridge.is_eeg_data_fresh()
    }

@app.post("/api/eeg/simulate")
async def simulate_eeg_data(request: dict):
    """Simulate receiving real EEG data from Mind Monitor (for testing)."""
    try:
        theta = request.get("theta", 15.0)
        alpha = request.get("alpha", 25.0)
        beta = request.get("beta", 20.0)
        gamma = request.get("gamma", 8.0)
        delta = request.get("delta", 20.0)
        
        realtime_bridge.receive_real_eeg_data(theta, alpha, beta, gamma, delta)
        
        return {
            "success": True,
            "message": "Real EEG data simulated",
            "data": {
                "theta": theta,
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "delta": delta
            },
            "data_status": "real_data"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates - streams data at 10Hz."""
    await websocket.accept()
    realtime_bridge.active_connections.append(websocket)
    print(f"WebSocket connection accepted - Total connections: {len(realtime_bridge.active_connections)}")
    
    try:
        while True:
            # Keep connection alive and handle any incoming messages
            try:
                data = await websocket.receive_text()
                # Handle any client messages if needed
            except:
                # No message received, continue
                pass
                
            # Connection is handled by background streaming task
            await asyncio.sleep(0.1)
            
    except WebSocketDisconnect:
        print(f"WebSocket disconnected - Total connections: {len(realtime_bridge.active_connections)}")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        # Remove from active connections
        if websocket in realtime_bridge.active_connections:
            realtime_bridge.active_connections.remove(websocket)
        try:
            await websocket.close()
        except:
            pass

if __name__ == "__main__":
    print("⚡ Real-Time EEG Data Bridge Starting...")
    print("🔄 Streaming at 10Hz (100ms intervals) for true real-time updates")
    print("📡 API Server: http://localhost:8001")
    print("🎧 EEG Analyzer: Port 9000")
    print("🔌 WebSocket: ws://localhost:8001/ws")
    print("=" * 70)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        ws="websockets"
    )
