#!/usr/bin/env python3
"""
MASTER EEG STRESS DETECTION SYSTEM
==================================

The ultimate consolidation of all EEG stress detection capabilities:
- Session-based analysis (30/60/90/120 second windows)
- Real-time streaming predictions  
- Advanced feature engineering with 18+ indicators
- XGBoost ML classifier with fallback algorithms
- Multi-window progressive confidence system
- OSC integration for Muse headbands via Mind Monitor

This single file contains the best of all previous implementations.

Usage:
    python master_eeg_analyzer.py --mode session    # Session analysis
    python master_eeg_analyzer.py --mode realtime   # Real-time streaming
    python master_eeg_analyzer.py --mode demo       # Test with simulated data

Author: Satyam Palkar
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from threading import Thread
import signal
import math
import statistics
from enum import Enum

# Core ML/Analysis imports
try:
    import torch
    import torch.nn as nn
    import xgboost as xgb
    TORCH_AVAILABLE = True
    XGB_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ ML libraries not available: {e}")
    TORCH_AVAILABLE = False
    XGB_AVAILABLE = False

# OSC import
try:
    from pythonosc import dispatcher, osc_server
    OSC_AVAILABLE = True
except ImportError:
    print("⚠️ OSC library not available - install python-osc")
    OSC_AVAILABLE = False

# Matplotlib for optional visualization
try:
    import matplotlib.pyplot as plt
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False

# Suppress warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

class StressLevel(Enum):
    VERY_RELAXED = "Very Relaxed"
    RELAXED = "Relaxed"
    NEUTRAL = "Neutral"
    LIGHT_STRESS = "Light Stress"
    MODERATE_STRESS = "Moderate Stress" 
    HIGH_STRESS = "High Stress"
    VERY_HIGH_STRESS = "Very High Stress"

@dataclass
class EEGSample:
    """Single EEG sample with metadata."""
    timestamp: datetime
    elapsed: float
    theta: float
    alpha: float
    beta: float
    gamma: float
    delta: float
    total_power: float
    valid: bool

@dataclass 
class StressMetrics:
    """Comprehensive stress analysis results."""
    overall_stress: float
    stress_level: str
    confidence: float
    arousal_level: float
    temporal_trend: str
    stress_indicators: Dict[str, float]
    recommendations: List[str]
    session_evidence: List[str]

@dataclass
class SessionResult:
    """Results for a complete session analysis."""
    window_seconds: int
    sample_count: int
    dominant_state: str
    stress_level: str
    confidence: float
    beta_alpha_ratio: float
    stress_index: float
    temporal_trend: str
    evidence: List[str]
    recommendations: List[str]
    raw_metrics: Dict[str, float]

# ============================================================================
# ADVANCED FEATURE ENGINEERING
# ============================================================================

class AdvancedFeatureEngineer:
    """
    Advanced EEG feature engineering with research-backed stress indicators.
    Handles NaN values, performs baseline calibration, and provides 18+ features.
    """
    
    def __init__(self, baseline_window: int = 120, adaptation_rate: float = 0.1):
        self.baseline_window = baseline_window
        self.adaptation_rate = adaptation_rate
        
        # Baseline statistics
        self.baseline_stats = {
            'theta_mean': 15.0, 'theta_std': 5.0,
            'alpha_mean': 25.0, 'alpha_std': 8.0, 
            'beta_mean': 20.0, 'beta_std': 6.0,
            'gamma_mean': 8.0, 'gamma_std': 3.0,
            'delta_mean': 30.0, 'delta_std': 10.0
        }
        
        self.baseline_buffer = deque(maxlen=baseline_window)
        self.calibrated = False
        self.sample_count = 0
        
        # Stress thresholds (research-backed, bias-corrected)
        self.stress_thresholds = {
            'very_relaxed': -0.3,
            'relaxed': -0.1,
            'neutral': 0.2,
            'light_stress': 0.5,
            'moderate_stress': 0.8,
            'high_stress': 1.2
        }
    
    def is_valid_eeg_data(self, theta: float, alpha: float, beta: float, 
                         gamma: float, delta: float) -> bool:
        """Comprehensive EEG data validation."""
        
        # Check for NaN or infinite values
        values = [theta, alpha, beta, gamma, delta]
        if any(not math.isfinite(x) for x in values):
            return False
            
        # Check for zero or negative values
        if any(x <= 0 for x in values):
            return False
            
        # Check for realistic ranges (based on Muse specifications)
        if not (1 <= theta <= 100): return False
        if not (1 <= alpha <= 150): return False  
        if not (1 <= beta <= 100): return False
        if not (1 <= gamma <= 80): return False
        if not (5 <= delta <= 200): return False
            
        # Check total power range
        total_power = sum(values)
        if total_power < 20 or total_power > 500:
            return False
            
        return True
    
    def safe_divide(self, numerator: float, denominator: float, default: float = 0.0) -> float:
        """Safe division with NaN/infinity handling."""
        if not math.isfinite(numerator) or not math.isfinite(denominator):
            return default
        if abs(denominator) < 1e-10:
            return default
        
        result = numerator / denominator
        return result if math.isfinite(result) else default
    
    def calculate_comprehensive_features(self, theta: float, alpha: float, beta: float,
                                       gamma: float, delta: float) -> Dict[str, float]:
        """Calculate 18+ research-backed EEG stress features."""
        
        if not self.is_valid_eeg_data(theta, alpha, beta, gamma, delta):
            return self._get_neutral_features()
        
        total_power = theta + alpha + beta + gamma + delta
        
        # Relative band powers
        rel_theta = theta / total_power
        rel_alpha = alpha / total_power
        rel_beta = beta / total_power  
        rel_gamma = gamma / total_power
        rel_delta = delta / total_power
        
        features = {}
        
        # 1-4: Relative powers
        features['rel_theta'] = rel_theta
        features['rel_alpha'] = rel_alpha
        features['rel_beta'] = rel_beta
        features['rel_gamma'] = rel_gamma
        
        # 5-8: Traditional ratios
        features['theta_beta_ratio'] = self.safe_divide(theta, beta, 1.0)
        features['alpha_beta_ratio'] = self.safe_divide(alpha, beta, 1.0)
        features['beta_alpha_ratio'] = self.safe_divide(beta, alpha, 1.0)
        features['gamma_beta_ratio'] = self.safe_divide(gamma, beta, 0.5)
        
        # 9-13: Advanced stress indicators
        stress_numerator = beta + gamma * 1.5
        stress_denominator = alpha + theta * 0.5
        features['stress_index'] = self.safe_divide(stress_numerator, stress_denominator, 1.0) - 1.0
        
        features['relaxation_index'] = self.safe_divide(alpha * 2.0, beta + gamma, 1.0)
        features['arousal_index'] = (rel_beta + rel_gamma * 1.5)
        features['focus_index'] = max(0, 1.0 - abs(features['beta_alpha_ratio'] - 1.2) * 0.5)
        features['drowsiness_index'] = self.safe_divide(theta + delta, beta + gamma, 1.0)
        
        # 14-16: Spectral features
        features['high_freq_power'] = beta + gamma
        features['low_freq_power'] = theta + delta
        features['spectral_balance'] = self.safe_divide(features['high_freq_power'], features['low_freq_power'], 1.0)
        
        # 17-18: Advanced indicators
        features['sympathetic_activation'] = self.safe_divide(beta + gamma, alpha + theta, 1.0)
        features['total_power'] = total_power
        
        # Update baseline if needed
        self._update_baseline(theta, alpha, beta, gamma, delta)
        
        return features
    
    def _update_baseline(self, theta: float, alpha: float, beta: float, gamma: float, delta: float):
        """Update baseline statistics for personalization."""
        sample = {'theta': theta, 'alpha': alpha, 'beta': beta, 'gamma': gamma, 'delta': delta}
        self.baseline_buffer.append(sample)
        self.sample_count += 1
        
        if len(self.baseline_buffer) >= 30 and self.sample_count % 10 == 0:  # Every 10 samples
            self._recalculate_baseline()
    
    def _recalculate_baseline(self):
        """Recalculate baseline using robust statistics."""
        if len(self.baseline_buffer) < 10:
            return
            
        data = {band: [s[band] for s in self.baseline_buffer] for band in ['theta', 'alpha', 'beta', 'gamma', 'delta']}
        
        for band in data:
            values = np.array(data[band])
            median = np.median(values)
            q75, q25 = np.percentile(values, [75, 25])
            iqr_std = (q75 - q25) / 1.349
            
            # Smooth update
            alpha = self.adaptation_rate
            self.baseline_stats[f'{band}_mean'] = alpha * median + (1 - alpha) * self.baseline_stats[f'{band}_mean']
            self.baseline_stats[f'{band}_std'] = alpha * max(iqr_std, 1.0) + (1 - alpha) * self.baseline_stats[f'{band}_std']
        
        if self.sample_count >= 50:
            self.calibrated = True
    
    def _get_neutral_features(self) -> Dict[str, float]:
        """Return neutral features for invalid data."""
        return {
            'rel_theta': 0.15, 'rel_alpha': 0.26, 'rel_beta': 0.20, 'rel_gamma': 0.08,
            'theta_beta_ratio': 0.75, 'alpha_beta_ratio': 1.25, 'beta_alpha_ratio': 0.8, 'gamma_beta_ratio': 0.4,
            'stress_index': 0.0, 'relaxation_index': 1.0, 'arousal_index': 0.2, 'focus_index': 0.5, 'drowsiness_index': 0.8,
            'high_freq_power': 28.0, 'low_freq_power': 45.0, 'spectral_balance': 0.62,
            'sympathetic_activation': 1.0, 'total_power': 98.0
        }

# ============================================================================
# NEURAL NETWORK MODEL (BiLSTM)
# ============================================================================

class EEGClassifier(nn.Module):
    """CNN-BiLSTM with Attention for EEG emotion classification."""
    
    def __init__(self, input_features: int = 1, hidden_size: int = 64, num_classes: int = 4):
        super().__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv1d(input_features, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        self.lstm = nn.LSTM(
            input_size=32, 
            hidden_size=hidden_size, 
            bidirectional=True, 
            batch_first=True
        )
        
        # Use the original saved model layer names
        self.attn = nn.Sequential()
        self.attn.add_module('attn', nn.Linear(hidden_size * 2, 1))
        
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        
        # Simple attention mechanism matching saved model
        attn_weights = torch.softmax(self.attn.attn(lstm_out), dim=1)
        attended_output = torch.sum(attn_weights * lstm_out, dim=1)
        
        return self.fc(attended_output)

# ============================================================================
# XGBOOST CLASSIFIER
# ============================================================================

class XGBoostStressClassifier:
    """XGBoost-based stress classifier with advanced features."""
    
    def __init__(self):
        self.model = None
        self.feature_names = [
            'rel_theta', 'rel_alpha', 'rel_beta', 'rel_gamma',
            'theta_beta_ratio', 'alpha_beta_ratio', 'beta_alpha_ratio', 'gamma_beta_ratio',
            'stress_index', 'relaxation_index', 'arousal_index', 'focus_index', 'drowsiness_index',
            'high_freq_power', 'low_freq_power', 'spectral_balance',
            'sympathetic_activation', 'total_power'
        ]
        
        self.label_encoder = {
            0: "Very Relaxed", 1: "Relaxed", 2: "Neutral",
            3: "Light Stress", 4: "Moderate Stress", 5: "High Stress"
        }
        
        self._create_model()
    
    def _create_model(self):
        """Create and train XGBoost model with synthetic data."""
        if not XGB_AVAILABLE:
            return
            
        # Generate synthetic training data
        X_train, y_train = self._generate_training_data()
        
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
        
        self.model.fit(X_train, y_train)
    
    def _generate_training_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """Generate realistic synthetic EEG training data."""
        fe = AdvancedFeatureEngineer()
        data_samples = []
        labels = []
        
        # EEG patterns for each stress level
        patterns = {
            0: (12, 45, 15, 5, 30),   # Very Relaxed
            1: (15, 40, 18, 6, 28),   # Relaxed
            2: (18, 30, 25, 8, 25),   # Neutral
            3: (12, 25, 35, 12, 20),  # Light Stress
            4: (10, 20, 45, 18, 15),  # Moderate Stress
            5: (8, 15, 55, 25, 12)    # High Stress
        }
        
        for stress_level, (t_base, a_base, b_base, g_base, d_base) in patterns.items():
            for _ in range(200):  # 200 samples per class
                # Add realistic noise
                theta = max(1, np.random.normal(t_base, 3))
                alpha = max(1, np.random.normal(a_base, 5))
                beta = max(1, np.random.normal(b_base, 4))
                gamma = max(1, np.random.normal(g_base, 2))
                delta = max(1, np.random.normal(d_base, 4))
                
                features = fe.calculate_comprehensive_features(theta, alpha, beta, gamma, delta)
                if features.get('total_power', 0) > 20:  # Valid sample
                    feature_vector = [features[name] for name in self.feature_names]
                    data_samples.append(feature_vector)
                    labels.append(stress_level)
        
        return pd.DataFrame(data_samples, columns=self.feature_names), np.array(labels)
    
    def predict(self, features: Dict[str, float]) -> Tuple[str, float]:
        """Predict stress level from features."""
        if not self.model:
            return "Model Not Available", 0.0
            
        feature_vector = [features.get(name, 0.0) for name in self.feature_names]
        feature_df = pd.DataFrame([feature_vector], columns=self.feature_names)
        
        probabilities = self.model.predict_proba(feature_df)[0]
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]
        
        return self.label_encoder[predicted_class], confidence

# ============================================================================
# MASTER EEG ANALYZER 
# ============================================================================

class MasterEEGAnalyzer:
    """
    The ultimate EEG stress detection system combining all approaches:
    - Session-based analysis with temporal integration
    - Real-time streaming with progressive confidence
    - Advanced feature engineering with bias correction
    - Multiple ML models with intelligent fallbacks
    """
    
    def __init__(self, mode: str = "session"):
        self.mode = mode
        self.osc_port = 8000
        
        # Core components
        self.feature_engineer = AdvancedFeatureEngineer()
        self.xgb_classifier = XGBoostStressClassifier()
        
        # Data storage
        self.session_data: List[EEGSample] = []
        self.realtime_buffer = deque(maxlen=30)  # 30-second rolling buffer
        
        # Session tracking
        self.start_time: Optional[datetime] = None
        self.is_collecting = False
        
        # Analysis results
        self.session_results: Dict[int, SessionResult] = {}
        
        # Initialize neural network if available
        self.neural_model = None
        if TORCH_AVAILABLE:
            try:
                self.neural_model = EEGClassifier()
                # Try to load pre-trained model
                if os.path.exists("emotion_model_new.pth"):
                    self.neural_model.load_state_dict(torch.load("emotion_model_new.pth", map_location='cpu'))
                    self.neural_model.eval()
            except Exception as e:
                print(f"⚠️ Neural model not loaded: {e}")
        
        print(f"🧠 Master EEG Analyzer initialized in {mode} mode")
        print(f"🔧 Features: XGBoost✅ Neural{'✅' if self.neural_model else '❌'} OSC{'✅' if OSC_AVAILABLE else '❌'}")
        
    def add_eeg_sample(self, theta: float, alpha: float, beta: float, 
                      gamma: float, delta: float) -> Optional[EEGSample]:
        """Add and validate EEG sample."""
        
        # Validate data
        if not self.feature_engineer.is_valid_eeg_data(theta, alpha, beta, gamma, delta):
            return None
            
        # Create sample
        now = datetime.now()
        elapsed = (now - self.start_time).total_seconds() if self.start_time else 0
        
        sample = EEGSample(
            timestamp=now,
            elapsed=elapsed, 
            theta=theta,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            total_power=theta + alpha + beta + gamma + delta,
            valid=True
        )
        
        # Store in appropriate buffers
        if self.is_collecting:
            self.session_data.append(sample)
            
        self.realtime_buffer.append(sample)
        
        return sample
    
    def analyze_realtime(self) -> StressMetrics:
        """Analyze current stress state for real-time mode."""
        
        if len(self.realtime_buffer) < 5:
            return StressMetrics(
                overall_stress=0.0,
                stress_level="Insufficient Data",
                confidence=0.0,
                arousal_level=0.0,
                temporal_trend="Unknown",
                stress_indicators={},
                recommendations=["Collecting data..."],
                session_evidence=[]
            )
        
        # Get recent samples for analysis
        recent_samples = list(self.realtime_buffer)[-10:]  # Last 10 seconds
        
        # Calculate average features
        avg_theta = np.mean([s.theta for s in recent_samples])
        avg_alpha = np.mean([s.alpha for s in recent_samples])
        avg_beta = np.mean([s.beta for s in recent_samples])
        avg_gamma = np.mean([s.gamma for s in recent_samples])
        avg_delta = np.mean([s.delta for s in recent_samples])
        
        # Extract features
        features = self.feature_engineer.calculate_comprehensive_features(
            avg_theta, avg_alpha, avg_beta, avg_gamma, avg_delta
        )
        
        # Get predictions from available models
        predictions = []
        
        # XGBoost prediction
        if self.xgb_classifier.model:
            xgb_pred, xgb_conf = self.xgb_classifier.predict(features)
            predictions.append((xgb_pred, xgb_conf, "XGBoost"))
        
        # Fallback to feature-based classification
        stress_index = features.get('stress_index', 0.0)
        beta_alpha = features.get('beta_alpha_ratio', 1.0)
        
        if stress_index > 0.6:
            fallback_pred = "High Stress"
        elif stress_index > 0.3:
            fallback_pred = "Moderate Stress"
        elif stress_index > 0.0:
            fallback_pred = "Light Stress"
        else:
            fallback_pred = "Relaxed"
        
        fallback_conf = min(0.8, 0.5 + abs(stress_index) * 0.3)
        predictions.append((fallback_pred, fallback_conf, "Feature-Based"))
        
        # Choose best prediction (highest confidence)
        best_pred, best_conf, best_method = max(predictions, key=lambda x: x[1])
        
        # Calculate temporal trend
        if len(self.realtime_buffer) >= 20:
            recent_stress = [features.get('stress_index', 0.0) for features in 
                           [self.feature_engineer.calculate_comprehensive_features(s.theta, s.alpha, s.beta, s.gamma, s.delta) 
                            for s in list(self.realtime_buffer)[-10:]]]
            older_stress = [features.get('stress_index', 0.0) for features in
                          [self.feature_engineer.calculate_comprehensive_features(s.theta, s.alpha, s.beta, s.gamma, s.delta)
                           for s in list(self.realtime_buffer)[-20:-10]]]
            
            recent_avg = np.mean(recent_stress)
            older_avg = np.mean(older_stress)
            
            if recent_avg > older_avg + 0.1:
                trend = "Increasing"
            elif recent_avg < older_avg - 0.1:
                trend = "Decreasing"
            else:
                trend = "Stable"
        else:
            trend = "Unknown"
        
        # Generate recommendations
        recommendations = self._generate_realtime_recommendations(best_pred, best_conf)
        
        return StressMetrics(
            overall_stress=features.get('stress_index', 0.0),
            stress_level=f"{best_pred} ({best_method})",
            confidence=best_conf,
            arousal_level=features.get('arousal_index', 0.0),
            temporal_trend=trend,
            stress_indicators=features,
            recommendations=recommendations,
            session_evidence=[f"Based on {len(recent_samples)} recent samples"]
        )
    
    def analyze_session_window(self, window_seconds: int) -> Optional[SessionResult]:
        """Analyze a specific session window with comprehensive temporal logic."""
        
        # Get data for this window
        window_data = [s for s in self.session_data if s.elapsed <= window_seconds]
        
        if len(window_data) < 10:
            return None
        
        # Extract all EEG values for the window
        theta_vals = [s.theta for s in window_data]
        alpha_vals = [s.alpha for s in window_data]
        beta_vals = [s.beta for s in window_data]
        gamma_vals = [s.gamma for s in window_data]
        delta_vals = [s.delta for s in window_data]
        
        # SESSION-WIDE CALCULATIONS (not instantaneous!)
        
        # 1. Average band powers across entire session
        avg_theta = np.mean(theta_vals)
        avg_alpha = np.mean(alpha_vals)
        avg_beta = np.mean(beta_vals)
        avg_gamma = np.mean(gamma_vals)
        avg_delta = np.mean(delta_vals)
        
        # 2. Calculate comprehensive features for session
        session_features = self.feature_engineer.calculate_comprehensive_features(
            avg_theta, avg_alpha, avg_beta, avg_gamma, avg_delta
        )
        
        # 3. Session-wide Beta/Alpha ratio
        beta_alpha_ratio = session_features['beta_alpha_ratio']
        stress_index = session_features['stress_index']
        
        # 4. Temporal analysis - divide session into segments
        segment_count = min(5, len(window_data) // 10)  # At least 10 samples per segment
        segment_size = len(window_data) // segment_count
        
        stress_timeline = []
        for i in range(segment_count):
            start_idx = i * segment_size
            end_idx = (i + 1) * segment_size if i < segment_count - 1 else len(window_data)
            segment = window_data[start_idx:end_idx]
            
            if segment:
                seg_theta = np.mean([s.theta for s in segment])
                seg_alpha = np.mean([s.alpha for s in segment])
                seg_beta = np.mean([s.beta for s in segment])
                seg_gamma = np.mean([s.gamma for s in segment])
                seg_delta = np.mean([s.delta for s in segment])
                
                seg_features = self.feature_engineer.calculate_comprehensive_features(
                    seg_theta, seg_alpha, seg_beta, seg_gamma, seg_delta
                )
                stress_timeline.append(seg_features['stress_index'])
        
        # 5. Calculate temporal trend
        if len(stress_timeline) > 1:
            trend_slope = np.polyfit(range(len(stress_timeline)), stress_timeline, 1)[0]
            if trend_slope > 0.1:
                temporal_trend = "Increasing"
            elif trend_slope < -0.1:
                temporal_trend = "Decreasing"
            else:
                stress_var = np.var(stress_timeline)
                temporal_trend = "Variable" if stress_var > 0.1 else "Stable"
        else:
            temporal_trend = "Insufficient Data"
        
        # 6. Dominant State Analysis using logical rules
        
        # Alpha dominance analysis
        alpha_dominant_count = sum(1 for s in window_data if s.alpha > s.beta)
        alpha_dominance_ratio = alpha_dominant_count / len(window_data)
        
        # Beta dominance analysis
        beta_dominant_count = sum(1 for s in window_data if s.beta > s.alpha and s.beta > s.theta)
        beta_dominance_ratio = beta_dominant_count / len(window_data)
        
        # High stress periods
        high_stress_count = sum(1 for stress in stress_timeline if stress > 0.4)
        high_stress_ratio = high_stress_count / len(stress_timeline) if stress_timeline else 0
        
        # LOGICAL DECISION TREE for Dominant State
        if stress_index > 0.6 and high_stress_ratio > 0.6:
            dominant_state = "Predominantly Stressed"
        elif stress_index > 0.3 and beta_dominance_ratio > 0.7:
            dominant_state = "Elevated Stress/Arousal"
        elif stress_index < -0.1 and alpha_dominance_ratio > 0.7:
            dominant_state = "Predominantly Relaxed"
        elif 1.0 < beta_alpha_ratio < 1.4 and abs(stress_index) < 0.3:
            dominant_state = "Sustained Focus"
        elif beta_alpha_ratio > 2.0:
            dominant_state = "High Arousal/Stress"
        elif beta_alpha_ratio < 0.8:
            dominant_state = "Low Arousal/Drowsy"
        else:
            dominant_state = "Mixed/Variable State"
        
        # 7. Map to categorical stress level
        if stress_index < -0.3:
            stress_level = "Very Low Stress"
        elif stress_index < -0.1:
            stress_level = "Low Stress"
        elif stress_index < 0.2:
            stress_level = "Moderate Stress"  
        elif stress_index < 0.6:
            stress_level = "High Stress"
        else:
            stress_level = "Very High Stress"
        
        # 8. Calculate session confidence
        base_confidence = 0.5
        
        # Sample count factor
        sample_factor = min(0.3, len(window_data) / 100)
        base_confidence += sample_factor
        
        # Consistency factor
        if stress_timeline:
            consistency = 1.0 / (1.0 + np.var(stress_timeline))
            base_confidence += consistency * 0.2
        
        # Window duration factor
        duration_factor = min(0.1, window_seconds / 600)  # Up to 10 minutes
        base_confidence += duration_factor
        
        confidence = min(0.95, base_confidence)
        
        # 9. Generate evidence
        evidence = []
        
        if beta_alpha_ratio > 1.5:
            evidence.append(f"Beta/Alpha ratio {beta_alpha_ratio:.2f} indicates activation")
        elif beta_alpha_ratio < 0.9:
            evidence.append(f"Beta/Alpha ratio {beta_alpha_ratio:.2f} suggests relaxation")
            
        if stress_index > 0.3:
            evidence.append(f"Stress index {stress_index:.2f} shows sustained stress")
        elif stress_index < -0.1:
            evidence.append(f"Negative stress index {stress_index:.2f} indicates calm state")
            
        if alpha_dominance_ratio > 0.7:
            evidence.append(f"Alpha dominant in {alpha_dominance_ratio*100:.0f}% of session")
        elif beta_dominance_ratio > 0.7:
            evidence.append(f"Beta dominant in {beta_dominance_ratio*100:.0f}% of session")
        
        if temporal_trend != "Stable":
            evidence.append(f"Stress pattern: {temporal_trend} over session")
        
        # 10. Generate session-specific recommendations
        recommendations = self._generate_session_recommendations(dominant_state, stress_level, confidence)
        
        # 11. Compile raw metrics
        raw_metrics = {
            'avg_theta': avg_theta, 'avg_alpha': avg_alpha, 'avg_beta': avg_beta,
            'avg_gamma': avg_gamma, 'avg_delta': avg_delta,
            'alpha_dominance': alpha_dominance_ratio,
            'beta_dominance': beta_dominance_ratio,
            'high_stress_ratio': high_stress_ratio,
            'stress_variability': np.var(stress_timeline) if stress_timeline else 0
        }
        raw_metrics.update(session_features)
        
        return SessionResult(
            window_seconds=window_seconds,
            sample_count=len(window_data),
            dominant_state=dominant_state,
            stress_level=stress_level,
            confidence=confidence,
            beta_alpha_ratio=beta_alpha_ratio,
            stress_index=stress_index,
            temporal_trend=temporal_trend,
            evidence=evidence,
            recommendations=recommendations,
            raw_metrics=raw_metrics
        )
    
    def _generate_realtime_recommendations(self, stress_level: str, confidence: float) -> List[str]:
        """Generate real-time recommendations."""
        recommendations = []
        
        if confidence < 0.4:
            recommendations.append("⏱️ Collecting more data for reliable assessment")
            return recommendations
            
        if "High Stress" in stress_level:
            recommendations.extend([
                "🚨 High stress detected - consider immediate action",
                "🧘‍♀️ Try deep breathing exercises",
                "⏸️ Take a short break from current activity"
            ])
        elif "Moderate Stress" in stress_level:
            recommendations.extend([
                "⚠️ Moderate stress - monitor and manage",
                "🌱 Practice mindfulness techniques",
                "☕ Consider a brief pause"
            ])
        elif "Light Stress" in stress_level:
            recommendations.extend([
                "📝 Light stress detected - note potential triggers",
                "🌿 Practice relaxation techniques"
            ])
        else:
            recommendations.extend([
                "✅ Good mental state detected",
                "🎯 Continue current approach"
            ])
            
        return recommendations
    
    def _generate_session_recommendations(self, dominant_state: str, stress_level: str, confidence: float) -> List[str]:
        """Generate session-specific recommendations."""
        recommendations = []
        
        if confidence < 0.6:
            recommendations.append("⚠️ Low confidence - consider longer sessions for better analysis")
        
        if "Stressed" in dominant_state or "High Stress" in stress_level:
            recommendations.extend([
                "🚨 Session shows significant stress patterns",
                "🧘‍♀️ Implement stress management strategies",
                "⏸️ Consider breaks during similar activities",
                "🌿 Review environmental factors"
            ])
        elif "Relaxed" in dominant_state or "Low Stress" in stress_level:
            recommendations.extend([
                "✅ Positive session patterns detected",
                "🎯 Good mental state for this activity",
                "📚 Consider replicating conditions"
            ])
        elif "Focus" in dominant_state:
            recommendations.extend([
                "🎯 Good sustained focus achieved",
                "⚡ Optimal cognitive state detected",
                "⏱️ Monitor for fatigue over longer periods"
            ])
        else:
            recommendations.extend([
                "📊 Variable session patterns",
                "🔍 Consider factors affecting consistency",
                "📈 Track patterns over multiple sessions"
            ])
            
        return recommendations
    
    # ========================================================================
    # SESSION MODE METHODS
    # ========================================================================
    
    def start_session_collection(self, duration: int = 120):
        """Start session data collection."""
        self.session_data = []
        self.start_time = datetime.now()
        self.is_collecting = True
        
        print(f"\n🚀 SESSION STARTED - {self.start_time.strftime('%H:%M:%S')}")
        print(f"📊 Collecting data for {duration} seconds...")
        print("🎧 Connect Mind Monitor and start your activity")
        print("=" * 60)
    
    def complete_session_analysis(self):
        """Complete session and analyze all windows."""
        if not self.is_collecting:
            return
            
        self.is_collecting = False
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()
        
        print("=" * 60)
        print(f"✅ SESSION COMPLETED - {end_time.strftime('%H:%M:%S')}")
        print(f"📊 Duration: {total_duration:.1f}s | Samples: {len(self.session_data)}")
        print("=" * 60)
        
        # Analyze target windows
        windows = [30, 60, 90, 120]
        
        for window_seconds in windows:
            if total_duration >= window_seconds:
                result = self.analyze_session_window(window_seconds)
                if result:
                    self.session_results[window_seconds] = result
                    self._print_session_result(result)
        
        # Final comparison
        self._print_session_comparison()
    
    def _print_session_result(self, result: SessionResult):
        """Print detailed session result."""
        print(f"\n🎯 {result.window_seconds}-SECOND SESSION ANALYSIS")
        print("-" * 50)
        print(f"📊 Samples: {result.sample_count}")
        print(f"🧠 Dominant State: {result.dominant_state}")
        print(f"⚡ Stress Level: {result.stress_level}")
        print(f"📈 Confidence: {result.confidence:.3f}")
        print(f"🔢 Beta/Alpha: {result.beta_alpha_ratio:.2f}")
        print(f"📉 Stress Index: {result.stress_index:.2f}")
        print(f"📊 Temporal Trend: {result.temporal_trend}")
        
        if result.evidence:
            print("🔍 Evidence:")
            for evidence in result.evidence[:3]:
                print(f"  • {evidence}")
        
        if result.recommendations and result.confidence > 0.6:
            print("💡 Recommendations:")
            for rec in result.recommendations[:2]:
                print(f"  • {rec}")
    
    def _print_session_comparison(self):
        """Print comparison across all session windows."""
        if not self.session_results:
            return
            
        print("\n" + "=" * 70)
        print("📋 COMPREHENSIVE SESSION COMPARISON")
        print("=" * 70)
        
        print("📊 Cross-Window Summary:")
        print("-" * 40)
        
        for window in sorted(self.session_results.keys()):
            result = self.session_results[window]
            print(f"{window:3d}s: {result.dominant_state:<25} | {result.confidence:.3f}")
        
        # Show progression
        if len(self.session_results) > 1:
            stress_values = [self.session_results[w].stress_index for w in sorted(self.session_results.keys())]
            confidence_values = [self.session_results[w].confidence for w in sorted(self.session_results.keys())]
            
            print(f"\n📈 Session Progression:")
            print(f"Stress Range: {min(stress_values):.2f} to {max(stress_values):.2f}")
            print(f"Confidence Range: {min(confidence_values):.3f} to {max(confidence_values):.3f}")
        
        # Final assessment
        best_result = None
        for window in sorted(self.session_results.keys(), reverse=True):
            if self.session_results[window].confidence > 0.6:
                best_result = self.session_results[window]
                break
        
        if not best_result:
            best_result = self.session_results[max(self.session_results.keys())]
        
        print(f"\n🎯 FINAL ASSESSMENT ({best_result.window_seconds}s analysis):")
        print(f"State: {best_result.dominant_state}")
        print(f"Stress: {best_result.stress_level}")
        print(f"Confidence: {best_result.confidence:.3f}")
        
        print("=" * 70)
    
    # ========================================================================
    # OSC INTEGRATION METHODS
    # ========================================================================
    
    def osc_handler(self, unused_addr, *args):
        """Handle incoming OSC data from Mind Monitor."""
        if len(args) < 4:
            return
            
        theta, alpha, beta, gamma = args[:4]
        delta = args[4] if len(args) > 4 else 25.0
        
        sample = self.add_eeg_sample(theta, alpha, beta, gamma, delta)
        
        if sample:
            if self.mode == "realtime":
                self._print_realtime_update()
            elif self.mode == "session":
                self._print_session_progress()
    
    def _print_realtime_update(self):
        """Print real-time analysis update."""
        if len(self.realtime_buffer) % 10 == 0:  # Every 10 samples
            metrics = self.analyze_realtime()
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            print(f"[{timestamp}] 🧠 {metrics.stress_level}")
            print(f"  ├─ Confidence: {metrics.confidence:.3f}")
            print(f"  ├─ Stress Index: {metrics.overall_stress:.2f}")
            print(f"  ├─ Trend: {metrics.temporal_trend}")
            
            if metrics.recommendations and metrics.confidence > 0.5:
                print(f"  └─ {metrics.recommendations[0]}")
            print()
    
    def _print_session_progress(self):
        """Print session collection progress."""
        if len(self.session_data) % 15 == 0:  # Every 15 samples
            elapsed = self.session_data[-1].elapsed if self.session_data else 0
            print(f"📈 {elapsed:.0f}s: {len(self.session_data)} samples collected")
            
        # Auto-complete session
        if self.session_data and self.session_data[-1].elapsed >= 120:
            self.complete_session_analysis()
    
    def enable_debug(self):
        """Enable debug mode to show all OSC messages."""
        self.debug_mode = True
        print("🔍 Debug mode ENABLED - will show all OSC messages")
    
    def disable_debug(self):
        """Disable debug mode for clean output."""
        self.debug_mode = False
        print("🔇 Debug mode DISABLED - clean predictions only")
    
    def _debug_osc_handler(self, address, *args):
        """Debug handler - only shows messages when debug is enabled."""
        if hasattr(self, 'debug_mode') and self.debug_mode:
            if "eeg" in address or any(band in address for band in ["theta", "alpha", "beta", "gamma", "delta"]):
                print(f"🧠 EEG OSC: {address} -> {args[:5]}...")
            else:
                print(f"🔍 DEBUG OSC: {address} -> {args[:5]}...")  # Show first 5 values
    
    def handle_band_power(self, address, *args):
        """Handle individual band power messages and combine them."""
        if not hasattr(self, '_band_buffer'):
            self._band_buffer = {}
            
        if not args:
            return
            
        value = args[0]
        
        # Extract band name from address and scale to realistic EEG range
        if "theta" in address:
            self._band_buffer['theta'] = value * 100  # Scale to typical EEG range
        elif "alpha" in address:
            self._band_buffer['alpha'] = value * 100
        elif "beta" in address:
            self._band_buffer['beta'] = value * 100
        elif "gamma" in address:
            self._band_buffer['gamma'] = value * 100
        elif "delta" in address:
            self._band_buffer['delta'] = value * 100
            
        # If we have at least the main 4 bands, process as EEG data
        required_bands = ['theta', 'alpha', 'beta', 'gamma']
        if all(band in self._band_buffer for band in required_bands):
            delta = self._band_buffer.get('delta', 25.0)  # Default delta
            
            # Only show processing info in debug mode
            if hasattr(self, 'debug_mode') and self.debug_mode:
                print(f"🎯 Processing EEG: θ={self._band_buffer['theta']:.1f}, α={self._band_buffer['alpha']:.1f}, β={self._band_buffer['beta']:.1f}, γ={self._band_buffer['gamma']:.1f}, δ={delta:.1f}")
            
            # Process the combined EEG data
            self.osc_handler("/muse/combined", 
                           self._band_buffer['theta'],
                           self._band_buffer['alpha'], 
                           self._band_buffer['beta'],
                           self._band_buffer['gamma'],
                           delta)
            
            # Reset buffer for next set
            self._band_buffer = {}
    
    def run_osc_session(self, duration: int = 120):
        """Run complete OSC session."""
        if not OSC_AVAILABLE:
            print("❌ OSC not available - install python-osc")
            return
            
        print(f"🎧 Starting {self.mode} mode on port {self.osc_port}")
        print("📡 Connect Mind Monitor and start collecting data")
        print("🛑 Press Ctrl+C to stop\n")
        
        # Setup signal handler
        def signal_handler(signum, frame):
            print("\n🛑 Stopping...")
            if self.mode == "session" and self.is_collecting:
                self.complete_session_analysis()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        # Setup OSC - listen to EEG addresses Mind Monitor uses
        osc_dispatcher = dispatcher.Dispatcher()
        # Main EEG addresses
        osc_dispatcher.map("/muse/eeg", self.osc_handler)
        osc_dispatcher.map("/muse/elements/raw_fft0", self.osc_handler) 
        osc_dispatcher.map("/muse/elements/raw_fft1", self.osc_handler)
        osc_dispatcher.map("/muse/elements/raw_fft2", self.osc_handler)
        osc_dispatcher.map("/muse/elements/raw_fft3", self.osc_handler)
        # Band power messages - these are what you're actually receiving
        osc_dispatcher.map("/muse/elements/theta_absolute", self.handle_band_power)
        osc_dispatcher.map("/muse/elements/alpha_absolute", self.handle_band_power)
        osc_dispatcher.map("/muse/elements/beta_absolute", self.handle_band_power)
        osc_dispatcher.map("/muse/elements/gamma_absolute", self.handle_band_power)
        osc_dispatcher.map("/muse/elements/delta_absolute", self.handle_band_power)
        # Alternative addresses
        osc_dispatcher.map("/muse/elements/theta_relative", self.handle_band_power)
        osc_dispatcher.map("/muse/elements/alpha_relative", self.handle_band_power)
        osc_dispatcher.map("/muse/elements/beta_relative", self.handle_band_power)
        osc_dispatcher.map("/muse/elements/gamma_relative", self.handle_band_power)
        osc_dispatcher.map("/muse/elements/delta_relative", self.handle_band_power)
        
        osc_dispatcher.set_default_handler(self._debug_osc_handler)
        
        server = osc_server.ThreadingOSCUDPServer(("0.0.0.0", self.osc_port), osc_dispatcher)
        
        if self.mode == "session":
            self.start_session_collection(duration)
        
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.shutdown()
    
    # ========================================================================
    # DEMO MODE METHODS  
    # ========================================================================
    
    def run_demo(self):
        """Run demo with simulated data."""
        print("🧪 Running Demo with Simulated EEG Data")
        print("=" * 50)
        
        if self.mode == "session":
            self._demo_session()
        else:
            self._demo_realtime()
    
    def _demo_session(self):
        """Demo session analysis with simulated data."""
        self.start_session_collection(120)
        
        # Simulate 120 seconds of changing EEG patterns
        patterns = [
            (30, "Relaxed", 15, 40, 18, 6, 28),
            (30, "Transition", 12, 30, 25, 10, 22), 
            (30, "Focused", 10, 25, 35, 12, 18),
            (30, "Stressed", 8, 15, 50, 20, 12)
        ]
        
        sample_count = 0
        for duration, phase, t_base, a_base, b_base, g_base, d_base in patterns:
            print(f"📊 Simulating {phase} phase ({duration}s)...")
            
            for second in range(duration):
                # Add realistic noise
                theta = max(1, np.random.normal(t_base, 3))
                alpha = max(1, np.random.normal(a_base, 5))
                beta = max(1, np.random.normal(b_base, 4))
                gamma = max(1, np.random.normal(g_base, 2))
                delta = max(1, np.random.normal(d_base, 4))
                
                self.add_eeg_sample(theta, alpha, beta, gamma, delta)
                sample_count += 1
                
                # Show progress
                if sample_count % 20 == 0:
                    print(f"  📈 {second + sum(p[0] for p in patterns[:patterns.index((duration, phase, t_base, a_base, b_base, g_base, d_base))]):.0f}s: {sample_count} samples")
                
                time.sleep(0.01)  # Fast simulation
        
        self.complete_session_analysis()
    
    def _demo_realtime(self):
        """Demo real-time analysis."""
        print("🔄 Real-time Demo - Simulating changing mental states...")
        
        states = [
            ("Relaxed", 15, 40, 18, 6, 28, 20),
            ("Focusing", 12, 30, 30, 10, 22, 15),  
            ("Stressed", 10, 20, 45, 18, 15, 25),
            ("Calming", 14, 35, 25, 8, 25, 20)
        ]
        
        for state_name, t, a, b, g, d, duration in states:
            print(f"\n📊 Simulating {state_name} state...")
            
            for i in range(duration):
                # Add noise
                theta = max(1, np.random.normal(t, 2))
                alpha = max(1, np.random.normal(a, 4))
                beta = max(1, np.random.normal(b, 3))
                gamma = max(1, np.random.normal(g, 2))
                delta = max(1, np.random.normal(d, 3))
                
                self.add_eeg_sample(theta, alpha, beta, gamma, delta)
                
                # Show analysis every 5 samples
                if i % 5 == 4:
                    metrics = self.analyze_realtime()
                    print(f"  🧠 {metrics.stress_level} (confidence: {metrics.confidence:.2f})")
                
                time.sleep(0.1)

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point with command line argument parsing."""
    
    parser = argparse.ArgumentParser(
        description="Master EEG Stress Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python master_eeg_analyzer.py --mode session    # Session analysis
  python master_eeg_analyzer.py --mode realtime   # Real-time streaming  
  python master_eeg_analyzer.py --mode demo       # Demo with simulated data
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["session", "realtime", "demo"],
        default="session",
        help="Analysis mode (default: session)"
    )
    
    parser.add_argument(
        "--port", 
        type=int,
        default=8000,
        help="OSC port for Mind Monitor (default: 8000)"
    )
    
    parser.add_argument(
        "--duration",
        type=int,
        default=120,
        help="Session duration in seconds (default: 120)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode to show all OSC messages"
    )
    
    args = parser.parse_args()
    
    # Display banner
    print("=" * 70)
    print("🧠 MASTER EEG STRESS DETECTION SYSTEM")
    print("🤖 Advanced Session Analysis + Real-time Intelligence")
    print("=" * 70)
    print(f"🎯 Mode: {args.mode.upper()}")
    print(f"📡 Port: {args.port}")
    if args.mode == "session":
        print(f"⏱️ Duration: {args.duration}s")
    print("=" * 70)
    
    # Initialize analyzer
    analyzer = MasterEEGAnalyzer(mode=args.mode)
    analyzer.osc_port = args.port
    
    # Enable debug mode if requested
    if args.debug:
        analyzer.enable_debug()
    
    # Run based on mode
    if args.mode == "demo":
        analyzer.run_demo()
    else:
        analyzer.run_osc_session(args.duration)

if __name__ == "__main__":
    main()
