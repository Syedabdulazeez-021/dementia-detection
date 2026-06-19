"""
Dementia Analyzer
Analyzes eye patterns and blink rate to assess dementia risk.
"""

import numpy as np
from collections import deque
import time
from datetime import datetime

class DementiaAnalyzer:
    """Analyzes eye patterns for dementia indicators."""
    
    def __init__(self, window_size=60):
        """
        Initialize analyzer.
        
        Args:
            window_size: Time window in seconds for analysis
        """
        self.window_size = window_size
        
        # Blink detection
        self.blink_history = deque(maxlen=1000)
        self.ear_threshold = 0.25  # EAR below this = closed eye
        self.blink_frames = deque(maxlen=1000)
        self.previous_state = None  # 'open' or 'closed'
        
        # Metrics tracking
        self.blink_count = 0
        self.session_start = time.time()
        self.ear_history = deque(maxlen=1000)
        self.predictions_history = deque(maxlen=1000)
        
        # Normal ranges (from research)
        self.NORMAL_BLINK_RATE_MIN = 12  # blinks per minute
        self.NORMAL_BLINK_RATE_MAX = 25
        self.NORMAL_EAR_MIN = 0.25
        self.NORMAL_EAR_MAX = 0.45
        
    def detect_blink(self, ear_left, ear_right, timestamp=None):
        """
        Detect if a blink occurred.
        
        Args:
            ear_left: Left eye aspect ratio
            ear_right: Right eye aspect ratio
            timestamp: Current timestamp
            
        Returns:
            True if blink detected, False otherwise
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Average EAR of both eyes
        avg_ear = (ear_left + ear_right) / 2.0
        
        # Store EAR history
        self.ear_history.append({
            'timestamp': timestamp,
            'ear': avg_ear,
            'left': ear_left,
            'right': ear_right
        })
        
        # Determine current state
        current_state = 'closed' if avg_ear < self.ear_threshold else 'open'
        
        # Detect blink (transition from open to closed)
        blink_detected = False
        if self.previous_state == 'open' and current_state == 'closed':
            blink_detected = True
            self.blink_count += 1
            self.blink_frames.append(timestamp)
            self.blink_history.append({
                'timestamp': timestamp,
                'ear': avg_ear
            })
        
        self.previous_state = current_state
        
        return blink_detected
    
    def calculate_blink_rate(self):
        """
        Calculate blinks per minute.
        
        Returns:
            Blink rate (float)
        """
        elapsed_time = time.time() - self.session_start
        
        if elapsed_time < 1:
            return 0.0
        
        # Blinks per minute
        blink_rate = (self.blink_count / elapsed_time) * 60
        
        return blink_rate
    
    def calculate_blink_variance(self):
        """
        Calculate variance in blink intervals (regularity).
        Higher variance = more irregular.
        
        Returns:
            Variance value (float)
        """
        if len(self.blink_frames) < 2:
            return 0.0
        
        # Calculate intervals between blinks
        intervals = []
        blink_times = list(self.blink_frames)
        
        for i in range(1, len(blink_times)):
            interval = blink_times[i] - blink_times[i-1]
            intervals.append(interval)
        
        if len(intervals) < 2:
            return 0.0
        
        variance = np.var(intervals)
        return variance
    
    def analyze_predictions(self, prediction_awake, prediction_sleepy):
        """
        Analyze model predictions over time.
        
        Args:
            prediction_awake: Probability of awake state
            prediction_sleepy: Probability of sleepy state
        """
        self.predictions_history.append({
            'timestamp': time.time(),
            'awake': prediction_awake,
            'sleepy': prediction_sleepy
        })
    
    def get_avg_prediction(self, window_seconds=10):
        """
        Get average prediction over recent window.
        
        Args:
            window_seconds: Time window in seconds
            
        Returns:
            Dictionary with average predictions
        """
        if not self.predictions_history:
            return {'awake': 0.5, 'sleepy': 0.5}
        
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        recent_predictions = [
            p for p in self.predictions_history 
            if p['timestamp'] >= cutoff_time
        ]
        
        if not recent_predictions:
            return {'awake': 0.5, 'sleepy': 0.5}
        
        avg_awake = np.mean([p['awake'] for p in recent_predictions])
        avg_sleepy = np.mean([p['sleepy'] for p in recent_predictions])
        
        return {
            'awake': avg_awake,
            'sleepy': avg_sleepy
        }
    
    def calculate_dementia_risk(self):
        """
        Calculate dementia risk score (0-100%).
        
        Based on research indicators:
        - Abnormal blink rate
        - Irregular blink patterns
        - High sleepy predictions
        - Low eye aspect ratio
        
        Returns:
            Risk score (0-100)
        """
        risk_score = 0
        risk_factors = []
        
        # 1. Blink Rate Analysis (30 points)
        blink_rate = self.calculate_blink_rate()
        
        if blink_rate < self.NORMAL_BLINK_RATE_MIN:
            deviation = (self.NORMAL_BLINK_RATE_MIN - blink_rate) / self.NORMAL_BLINK_RATE_MIN
            risk_score += min(30, deviation * 30)
            risk_factors.append(f"Low blink rate ({blink_rate:.1f} bpm)")
        elif blink_rate > self.NORMAL_BLINK_RATE_MAX:
            deviation = (blink_rate - self.NORMAL_BLINK_RATE_MAX) / self.NORMAL_BLINK_RATE_MAX
            risk_score += min(30, deviation * 30)
            risk_factors.append(f"High blink rate ({blink_rate:.1f} bpm)")
        
        # 2. Blink Pattern Regularity (25 points)
        variance = self.calculate_blink_variance()
        if variance > 2.0:  # Threshold for irregular patterns
            risk_score += min(25, (variance / 5.0) * 25)
            risk_factors.append(f"Irregular blink pattern (var: {variance:.2f})")
        
        # 3. Sleepy State Dominance (25 points)
        avg_pred = self.get_avg_prediction(window_seconds=30)
        if avg_pred['sleepy'] > 0.6:
            risk_score += (avg_pred['sleepy'] - 0.6) / 0.4 * 25
            risk_factors.append(f"High sleepy state ({avg_pred['sleepy']*100:.1f}%)")
        
        # 4. Low Eye Aspect Ratio (20 points)
        if self.ear_history:
            recent_ears = [e['ear'] for e in list(self.ear_history)[-100:]]
            avg_ear = np.mean(recent_ears)
            
            if avg_ear < self.NORMAL_EAR_MIN:
                deviation = (self.NORMAL_EAR_MIN - avg_ear) / self.NORMAL_EAR_MIN
                risk_score += min(20, deviation * 20)
                risk_factors.append(f"Low eye openness (EAR: {avg_ear:.3f})")
        
        # Cap at 100
        risk_score = min(100, risk_score)
        
        return {
            'risk_score': risk_score,
            'risk_level': self._get_risk_level(risk_score),
            'risk_factors': risk_factors,
            'blink_rate': blink_rate,
            'blink_variance': variance,
            'avg_predictions': avg_pred,
            'total_blinks': self.blink_count
        }
    
    def _get_risk_level(self, score):
        """Convert risk score to level."""
        if score < 20:
            return "LOW"
        elif score < 40:
            return "MILD"
        elif score < 60:
            return "MODERATE"
        elif score < 80:
            return "HIGH"
        else:
            return "VERY HIGH"
    
    def get_session_stats(self):
        """Get overall session statistics."""
        elapsed_time = time.time() - self.session_start
        
        stats = {
            'session_duration': elapsed_time,
            'total_blinks': self.blink_count,
            'blink_rate': self.calculate_blink_rate(),
            'blink_variance': self.calculate_blink_variance(),
            'total_frames': len(self.ear_history),
            'avg_ear': np.mean([e['ear'] for e in self.ear_history]) if self.ear_history else 0.0
        }
        
        return stats
    
    def reset(self):
        """Reset analyzer for new session."""
        self.blink_history.clear()
        self.blink_frames.clear()
        self.ear_history.clear()
        self.predictions_history.clear()
        self.blink_count = 0
        self.previous_state = None
        self.session_start = time.time()
    
    def export_session_data(self, filepath='session_data.csv'):
        """Export session data to CSV."""
        import csv
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(['Timestamp', 'EAR', 'Left_EAR', 'Right_EAR', 'Blink'])
            
            # Write data
            blink_times = set([b['timestamp'] for b in self.blink_history])
            
            for entry in self.ear_history:
                is_blink = 1 if entry['timestamp'] in blink_times else 0
                writer.writerow([
                    entry['timestamp'],
                    entry['ear'],
                    entry['left'],
                    entry['right'],
                    is_blink
                ])
        
        print(f"Session data exported to: {filepath}")


# Example usage
if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                    Dementia Analyzer - Test                          ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Simulate analysis
    analyzer = DementiaAnalyzer()
    
    print("\nSimulating 60 seconds of eye tracking...")
    
    # Simulate normal blinking
    for i in range(60):
        # Simulate EAR values
        if i % 5 == 0:  # Blink every 5 seconds
            ear_left = 0.15
            ear_right = 0.15
        else:
            ear_left = 0.35
            ear_right = 0.35
        
        analyzer.detect_blink(ear_left, ear_right)
        analyzer.analyze_predictions(0.7, 0.3)  # Mostly awake
        time.sleep(0.1)
    
    # Get risk assessment
    risk = analyzer.calculate_dementia_risk()
    
    print("\n" + "="*70)
    print("DEMENTIA RISK ASSESSMENT")
    print("="*70)
    print(f"Risk Score: {risk['risk_score']:.1f}%")
    print(f"Risk Level: {risk['risk_level']}")
    print(f"Blink Rate: {risk['blink_rate']:.1f} blinks/minute")
    print(f"Total Blinks: {risk['total_blinks']}")
    print(f"\nRisk Factors:")
    for factor in risk['risk_factors']:
        print(f"  - {factor}")
    print("="*70)
