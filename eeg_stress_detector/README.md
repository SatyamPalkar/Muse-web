# 🧠 EEG Stress Detector Package

**Advanced EEG stress detection system** with session-based temporal analysis and real-time monitoring.

## 🚀 Installation

```bash
pip install eeg-stress-detector
```

## ✨ Features

- **Session Analysis**: 30/60/90/120 second temporal integration
- **Real-time Monitoring**: Live stress detection with progressive confidence
- **Advanced ML**: XGBoost + BiLSTM neural network classifiers
- **Muse Integration**: Direct OSC compatibility via Mind Monitor
- **Feature Engineering**: 18+ research-backed EEG indicators
- **Bias Correction**: Personalized stress thresholds

## 🎯 Quick Start

### Basic Usage
```python
from eeg_stress_detector import StressDetector

# Initialize detector
detector = StressDetector()

# Analyze single sample
result = detector.analyze_single_sample(
    theta=15.0, alpha=25.0, beta=20.0, gamma=8.0, delta=30.0
)

print(f"Stress Level: {result['stress_level']}")
print(f"Confidence: {result['confidence']:.2f}")
```

### Session Analysis
```python
from eeg_stress_detector import SessionAnalyzer

# Initialize session analyzer
analyzer = SessionAnalyzer(target_windows=[30, 60, 90, 120])

# Start session
analyzer.start_session()

# Add EEG samples (from your data source)
for theta, alpha, beta, gamma, delta in eeg_data:
    analyzer.add_sample(theta, alpha, beta, gamma, delta)

# Analyze complete session
results = analyzer.analyze_session()
print(results)
```

### OSC Integration
```python
from eeg_stress_detector import OSCReceiver

# Connect to Mind Monitor
receiver = OSCReceiver(port=8000)
receiver.start_monitoring()
```

## 📊 Package Components

- `StressDetector`: Core stress analysis with feature engineering
- `SessionAnalyzer`: Temporal integration across multiple time windows  
- `OSCReceiver`: Mind Monitor integration for real-time data
- `utils`: Validation and helper functions

## 🎛️ Command Line Interface

```bash
# Start real-time monitoring
eeg-stress-detector --mode realtime --port 8000

# Run session analysis
eeg-stress-detector --mode session --duration 120

# Demo with simulated data
eeg-stress-detector --demo
```

## 🧮 Scientific Approach

Uses **session-wide temporal integration** rather than instantaneous readings:

```python
# Session Stress Index (consistency-weighted)
Raw_Stress = ((mean(Beta) + mean(Gamma)*1.5) / (mean(Alpha) + mean(Theta)*0.5)) - 1.0
Beta_Consistency = 1.0 / (1.0 + std(Beta) / (mean(Beta) + 0.01))
Final_Stress_Index = Raw_Stress × Beta_Consistency
```

## 📋 Requirements

- Python 3.8+
- NumPy >= 1.21.0
- PyTorch >= 1.9.0  
- XGBoost >= 1.5.0
- python-osc >= 1.8.0

## 🤝 Contributing

Contributions welcome! Please see the [GitHub repository](https://github.com/SatyamPalkar/eeg-stress-detector) for development guidelines.

## 📄 License

MIT License - see LICENSE file for details.

---

**🧠 Advanced EEG stress detection made simple!**