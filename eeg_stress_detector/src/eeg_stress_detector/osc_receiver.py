"""OSC Receiver for Mind Monitor integration."""

try:
    from pythonosc import dispatcher, osc_server
    OSC_AVAILABLE = True
except ImportError:
    OSC_AVAILABLE = False

from .stress_detector import StressDetector


class OSCReceiver:
    """OSC receiver for EEG data from Mind Monitor."""
    
    def __init__(self, port=8000):
        self.port = port
        self.detector = StressDetector()
        
    def start_monitoring(self):
        """Start OSC monitoring."""
        if not OSC_AVAILABLE:
            print("❌ OSC not available - install python-osc")
            return
        
        print(f"📡 Starting OSC receiver on port {self.port}")
        # Implementation would use pythonosc here


class MuseReceiver(OSCReceiver):
    """Specialized receiver for Muse headbands."""
    
    def __init__(self, port=8000):
        super().__init__(port)
        print("🎧 Muse receiver initialized")