"""Utility functions for EEG stress detection."""

import math
from typing import Dict


def validate_eeg_data(theta: float, alpha: float, beta: float, 
                     gamma: float, delta: float) -> bool:
    """Validate EEG data ranges and values."""
    
    values = [theta, alpha, beta, gamma, delta]
    
    # Check for NaN or infinite values
    if any(not math.isfinite(x) for x in values):
        return False
        
    # Check for realistic ranges
    if not (1 <= theta <= 100): return False
    if not (1 <= alpha <= 150): return False
    if not (1 <= beta <= 100): return False
    if not (1 <= gamma <= 80): return False
    if not (5 <= delta <= 200): return False
        
    return True


def calculate_stress_features(theta: float, alpha: float, beta: float,
                            gamma: float, delta: float) -> Dict[str, float]:
    """Calculate basic stress features from EEG bands."""
    
    if not validate_eeg_data(theta, alpha, beta, gamma, delta):
        return {}
    
    total_power = theta + alpha + beta + gamma + delta
    
    return {
        'beta_alpha_ratio': beta / (alpha + 0.01),
        'stress_index': (beta + gamma * 1.5) / (alpha + theta * 0.5 + 0.01) - 1.0,
        'total_power': total_power,
        'arousal_index': (beta + gamma) / total_power
    }