"""
EEG Stress Detector Package
==========================

Advanced EEG stress detection system with session-based temporal analysis,
real-time monitoring, and comprehensive machine learning integration.

Key Features:
- Session analysis with 30/60/90/120 second windows
- Real-time stress monitoring with progressive confidence
- XGBoost and BiLSTM neural network classifiers
- OSC integration for Muse headbands via Mind Monitor
- Advanced feature engineering with 18+ indicators
- Bias-corrected stress assessment algorithms

Author: Satyam Palkar
"""

__version__ = "1.0.1"
__author__ = "Satyam Palkar"
__email__ = "satyam@example.com"

from .stress_detector import StressDetector, SessionAnalyzer
from .osc_receiver import OSCReceiver, MuseReceiver
from .utils import validate_eeg_data, calculate_stress_features

__all__ = [
    "StressDetector",
    "SessionAnalyzer", 
    "OSCReceiver",
    "MuseReceiver",
    "validate_eeg_data",
    "calculate_stress_features",
    "__version__",
    "__author__",
    "__email__",
]

# Package metadata
PACKAGE_INFO = {
    "name": "eeg-stress-detector",
    "version": __version__,
    "description": "Advanced EEG stress detection system",
    "author": __author__,
    "author_email": __email__,
    "features": [
        "Session-based temporal analysis",
        "Real-time monitoring",
        "XGBoost + BiLSTM classification", 
        "OSC/Mind Monitor integration",
        "Advanced feature engineering",
        "Bias-corrected algorithms"
    ]
}

def get_package_info():
    """Get package information."""
    return PACKAGE_INFO.copy()

def print_welcome():
    """Print package welcome message."""
    print(f"""
🧠 EEG Stress Detector v{__version__}
===================================
Advanced stress detection with session analysis and real-time monitoring.

Features:
• Session-based temporal analysis (30/60/90/120s windows)
• Real-time monitoring with progressive confidence  
• XGBoost + BiLSTM neural network classifiers
• Direct Muse headband integration via Mind Monitor
• Research-backed feature engineering (18+ indicators)
• Bias-corrected stress assessment algorithms

Quick Start:
  from eeg_stress_detector import StressDetector
  detector = StressDetector()
  detector.start_session_analysis()

For more examples, see: examples/
    """)

# Print welcome message on import
if __name__ != "__main__":
    import os
    if os.getenv("EEG_STRESS_DETECTOR_QUIET") != "1":
        print_welcome()