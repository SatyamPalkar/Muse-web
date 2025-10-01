#!/usr/bin/env python3
"""
EEG API Server
==============

FastAPI server that exposes the MasterEEGAnalyzer as REST API endpoints.
Provides real-time EEG data, session analysis, and WebSocket streaming.

Usage:
    python eeg_api_server.py --port 8001 --mode realtime
    python eeg_api_server.py --port 8001 --mode session
"""

import os
import sys
import argparse
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
from threading import Thread, Event
import signal
import uvicorn

# FastAPI and WebSocket imports
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Import our EEG analyzer
from master_eeg_analyzer import MasterEEGAnalyzer, StressMetrics, SessionResult

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class EEGData(BaseModel):
    theta: float
    alpha: float
    beta: float
    gamma: float
    delta: float
    timestamp: Optional[str] = None

class AnalysisRequest(BaseModel):
    mode: str = "realtime"  # realtime, session
    duration: Optional[int] = 120  # For session mode
    port: Optional[int] = 8000  # OSC port

class SessionConfig(BaseModel):
    duration: int = 120
    auto_start: bool = True

class CommandRequest(BaseModel):
    command: str  # start, stop, reset, status
    data: Optional[Dict[str, Any]] = None

# ============================================================================
# EEG API SERVER
# ============================================================================

class EEGAPIServer:
    """FastAPI server for EEG stress detection system."""
    
    def __init__(self, port: int = 8001, mode: str = "realtime"):
        self.port = port
        self.mode = mode
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="EEG Stress Detection API",
            description="Real-time EEG analysis and stress detection",
            version="1.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Initialize EEG analyzer
        self.analyzer = MasterEEGAnalyzer(mode=mode)
        self.analyzer.osc_port = 8000
        
        # WebSocket connections
        self.websocket_connections: List[WebSocket] = []
        
        # Background tasks
        self.background_task = None
        self.stop_event = Event()
        
        # Setup routes
        self._setup_routes()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _setup_routes(self):
        """Setup all API routes."""
        
        @self.app.get("/")
        async def root():
            return {
                "message": "EEG Stress Detection API",
                "version": "1.0.0",
                "mode": self.mode,
                "endpoints": {
                    "realtime": "/api/realtime",
                    "session": "/api/session",
                    "websocket": "/ws",
                    "status": "/api/status",
                    "health": "/health"
                }
            }
        
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "analyzer_mode": self.mode,
                "websocket_connections": len(self.websocket_connections)
            }
        
        @self.app.get("/api/status")
        async def get_status():
            """Get current analyzer status."""
            return {
                "mode": self.mode,
                "is_collecting": self.analyzer.is_collecting,
                "samples_collected": len(self.analyzer.session_data),
                "realtime_buffer_size": len(self.analyzer.realtime_buffer),
                "session_duration": (datetime.now() - self.analyzer.start_time).total_seconds() if self.analyzer.start_time else 0,
                "osc_port": self.analyzer.osc_port,
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.post("/api/analyze")
        async def analyze_eeg_data(eeg_data: EEGData):
            """Analyze single EEG sample."""
            try:
                sample = self.analyzer.add_eeg_sample(
                    eeg_data.theta,
                    eeg_data.alpha,
                    eeg_data.beta,
                    eeg_data.gamma,
                    eeg_data.delta
                )
                
                if not sample:
                    raise HTTPException(status_code=400, detail="Invalid EEG data")
                
                # Get real-time analysis
                metrics = self.analyzer.analyze_realtime()
                
                return {
                    "success": True,
                    "data": {
                        "timestamp": sample.timestamp.isoformat(),
                        "stress_level": metrics.stress_level,
                        "confidence": metrics.confidence,
                        "overall_stress": metrics.overall_stress,
                        "arousal_level": metrics.arousal_level,
                        "temporal_trend": metrics.temporal_trend,
                        "recommendations": metrics.recommendations,
                        "stress_indicators": metrics.stress_indicators
                    }
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/realtime")
        async def get_realtime_analysis():
            """Get current real-time analysis."""
            try:
                if len(self.analyzer.realtime_buffer) < 5:
                    return {
                        "success": True,
                        "data": {
                            "status": "insufficient_data",
                            "message": "Need more data for reliable analysis",
                            "samples_needed": 5 - len(self.analyzer.realtime_buffer)
                        }
                    }
                
                metrics = self.analyzer.analyze_realtime()
                
                return {
                    "success": True,
                    "data": {
                        "timestamp": datetime.now().isoformat(),
                        "stress_level": metrics.stress_level,
                        "confidence": metrics.confidence,
                        "overall_stress": metrics.overall_stress,
                        "arousal_level": metrics.arousal_level,
                        "temporal_trend": metrics.temporal_trend,
                        "recommendations": metrics.recommendations,
                        "stress_indicators": metrics.stress_indicators,
                        "session_evidence": metrics.session_evidence,
                        "sample_count": len(self.analyzer.realtime_buffer)
                    }
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/session/start")
        async def start_session(config: SessionConfig):
            """Start a new session analysis."""
            try:
                if self.analyzer.is_collecting:
                    return {
                        "success": False,
                        "message": "Session already in progress",
                        "current_duration": (datetime.now() - self.analyzer.start_time).total_seconds() if self.analyzer.start_time else 0
                    }
                
                self.analyzer.start_session_collection(config.duration)
                
                return {
                    "success": True,
                    "message": f"Session started for {config.duration} seconds",
                    "start_time": self.analyzer.start_time.isoformat() if self.analyzer.start_time else None,
                    "duration": config.duration
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/session/stop")
        async def stop_session():
            """Stop current session and get results."""
            try:
                if not self.analyzer.is_collecting:
                    return {
                        "success": False,
                        "message": "No session in progress"
                    }
                
                self.analyzer.complete_session_analysis()
                
                # Convert session results to JSON-serializable format
                session_results = {}
                for window, result in self.analyzer.session_results.items():
                    session_results[str(window)] = {
                        "window_seconds": result.window_seconds,
                        "sample_count": result.sample_count,
                        "dominant_state": result.dominant_state,
                        "stress_level": result.stress_level,
                        "confidence": result.confidence,
                        "beta_alpha_ratio": result.beta_alpha_ratio,
                        "stress_index": result.stress_index,
                        "temporal_trend": result.temporal_trend,
                        "evidence": result.evidence,
                        "recommendations": result.recommendations,
                        "raw_metrics": result.raw_metrics
                    }
                
                return {
                    "success": True,
                    "message": "Session completed",
                    "total_samples": len(self.analyzer.session_data),
                    "session_duration": (datetime.now() - self.analyzer.start_time).total_seconds() if self.analyzer.start_time else 0,
                    "results": session_results
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/session/results")
        async def get_session_results():
            """Get current session results."""
            try:
                if not self.analyzer.session_results:
                    return {
                        "success": True,
                        "data": {
                            "message": "No session results available",
                            "is_collecting": self.analyzer.is_collecting
                        }
                    }
                
                # Convert to JSON-serializable format
                session_results = {}
                for window, result in self.analyzer.session_results.items():
                    session_results[str(window)] = {
                        "window_seconds": result.window_seconds,
                        "sample_count": result.sample_count,
                        "dominant_state": result.dominant_state,
                        "stress_level": result.stress_level,
                        "confidence": result.confidence,
                        "beta_alpha_ratio": result.beta_alpha_ratio,
                        "stress_index": result.stress_index,
                        "temporal_trend": result.temporal_trend,
                        "evidence": result.evidence,
                        "recommendations": result.recommendations,
                        "raw_metrics": result.raw_metrics
                    }
                
                return {
                    "success": True,
                    "data": session_results
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/command")
        async def send_command(command: CommandRequest):
            """Send commands to the analyzer."""
            try:
                if command.command == "start_osc":
                    # Start OSC server in background
                    def run_osc():
                        self.analyzer.run_osc_session(120)
                    
                    osc_thread = Thread(target=run_osc, daemon=True)
                    osc_thread.start()
                    
                    return {
                        "success": True,
                        "message": "OSC server started",
                        "port": self.analyzer.osc_port
                    }
                
                elif command.command == "enable_debug":
                    self.analyzer.enable_debug()
                    return {
                        "success": True,
                        "message": "Debug mode enabled"
                    }
                
                elif command.command == "disable_debug":
                    self.analyzer.disable_debug()
                    return {
                        "success": True,
                        "message": "Debug mode disabled"
                    }
                
                elif command.command == "reset":
                    self.analyzer.session_data = []
                    self.analyzer.realtime_buffer.clear()
                    self.analyzer.session_results = {}
                    self.analyzer.is_collecting = False
                    self.analyzer.start_time = None
                    
                    return {
                        "success": True,
                        "message": "Analyzer reset"
                    }
                
                else:
                    raise HTTPException(status_code=400, detail=f"Unknown command: {command.command}")
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time data streaming."""
            await websocket.accept()
            self.websocket_connections.append(websocket)
            
            try:
                # Send initial status
                await websocket.send_text(json.dumps({
                    "type": "status",
                    "data": {
                        "message": "Connected to EEG API",
                        "mode": self.mode,
                        "timestamp": datetime.now().isoformat()
                    }
                }))
                
                # Keep connection alive and send periodic updates
                while True:
                    try:
                        # Send real-time analysis if available
                        if len(self.analyzer.realtime_buffer) >= 5:
                            metrics = self.analyzer.analyze_realtime()
                            
                            data = {
                                "type": "realtime_analysis",
                                "data": {
                                    "timestamp": datetime.now().isoformat(),
                                    "stress_level": metrics.stress_level,
                                    "confidence": metrics.confidence,
                                    "overall_stress": metrics.overall_stress,
                                    "arousal_level": metrics.arousal_level,
                                    "temporal_trend": metrics.temporal_trend,
                                    "recommendations": metrics.recommendations,
                                    "stress_indicators": metrics.stress_indicators,
                                    "sample_count": len(self.analyzer.realtime_buffer)
                                }
                            }
                            
                            await websocket.send_text(json.dumps(data))
                        
                        # Send session updates if collecting
                        if self.analyzer.is_collecting and len(self.analyzer.session_data) % 15 == 0:
                            session_data = {
                                "type": "session_update",
                                "data": {
                                    "timestamp": datetime.now().isoformat(),
                                    "samples_collected": len(self.analyzer.session_data),
                                    "elapsed_time": (datetime.now() - self.analyzer.start_time).total_seconds() if self.analyzer.start_time else 0,
                                    "is_collecting": self.analyzer.is_collecting
                                }
                            }
                            await websocket.send_text(json.dumps(session_data))
                        
                        # Wait before next update
                        await asyncio.sleep(2)  # Update every 2 seconds
                        
                    except WebSocketDisconnect:
                        break
                        
            except WebSocketDisconnect:
                pass
            finally:
                if websocket in self.websocket_connections:
                    self.websocket_connections.remove(websocket)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print(f"\n🛑 Received signal {signum}, shutting down...")
        self.stop_event.set()
        sys.exit(0)
    
    async def start_background_tasks(self):
        """Start background tasks for OSC and data processing."""
        if self.mode == "realtime":
            # Start OSC server in background for real-time mode
            def run_osc():
                try:
                    self.analyzer.run_osc_session(120)
                except Exception as e:
                    print(f"OSC server error: {e}")
            
            osc_thread = Thread(target=run_osc, daemon=True)
            osc_thread.start()
            print(f"🎧 OSC server started on port {self.analyzer.osc_port}")
    
    def run(self, host: str = "0.0.0.0"):
        """Run the FastAPI server."""
        print("=" * 70)
        print("🧠 EEG STRESS DETECTION API SERVER")
        print("=" * 70)
        print(f"🌐 Server: http://{host}:{self.port}")
        print(f"📊 Mode: {self.mode.upper()}")
        print(f"🎧 OSC Port: {self.analyzer.osc_port}")
        print("=" * 70)
        print("📡 Available endpoints:")
        print(f"  • API Docs: http://{host}:{self.port}/docs")
        print(f"  • Health: http://{host}:{self.port}/health")
        print(f"  • Realtime: http://{host}:{self.port}/api/realtime")
        print(f"  • WebSocket: ws://{host}:{self.port}/ws")
        print("=" * 70)
        
        # Start background tasks
        asyncio.create_task(self.start_background_tasks())
        
        # Run server
        uvicorn.run(
            self.app,
            host=host,
            port=self.port,
            log_level="info"
        )

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="EEG Stress Detection API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="API server port (default: 8001)"
    )
    
    parser.add_argument(
        "--mode",
        choices=["realtime", "session"],
        default="realtime",
        help="Analyzer mode (default: realtime)"
    )
    
    parser.add_argument(
        "--osc-port",
        type=int,
        default=8000,
        help="OSC port for Mind Monitor (default: 8000)"
    )
    
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Server host (default: 0.0.0.0)"
    )
    
    args = parser.parse_args()
    
    # Create and run server
    server = EEGAPIServer(port=args.port, mode=args.mode)
    server.analyzer.osc_port = args.osc_port
    server.run(host=args.host)

if __name__ == "__main__":
    main()
