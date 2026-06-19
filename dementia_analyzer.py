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
        
        # Adaptive blink detection
        self.blink_history = deque(maxlen=1000)
        self.ear_threshold = 0.30  # Initial threshold, will be calibrated
        self.blink_frames = deque(maxlen=1000)
        self.previous_state = None  # 'open' or 'closed'
        
        # Micro-sleep detection
        self.micro_sleeps = []
        self.current_blink_start = None
        self.partial_blinks = []
        self.partial_threshold_start = 0.30
        self.current_dip_min_ear = 1.0
        self.in_dip = False
        
        # Calibration system
        self.is_calibrated = False
        self.calibration_samples = []
        self.calibration_duration = 5.0  # Calibrate for 5 seconds
        self.calibration_start_time = None
        self.baseline_ear = None
        
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
        Detect if a blink occurred with adaptive threshold.
        
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
        
        # Calibration phase
        if not self.is_calibrated:
            if self.calibration_start_time is None:
                self.calibration_start_time = timestamp
            
            # Collect samples during calibration period
            elapsed = timestamp - self.calibration_start_time
            
            if elapsed < self.calibration_duration:
                # Still calibrating - collect samples
                self.calibration_samples.append(avg_ear)
                
                # Store EAR history but don't detect blinks yet
                self.ear_history.append({
                    'timestamp': timestamp,
                    'ear': avg_ear,
                    'left': ear_left,
                    'right': ear_right
                })
                
                return False  # No blink detection during calibration
            else:
                # Calibration complete - set adaptive threshold
                if self.calibration_samples:
                    # Use median of top 80% samples (ignore blinks during calibration)
                    sorted_samples = sorted(self.calibration_samples)
                    top_80_percent = sorted_samples[int(len(sorted_samples) * 0.2):]
                    self.baseline_ear = np.median(top_80_percent)
                    
                    # Set threshold to 85% of baseline
                    self.ear_threshold = self.baseline_ear * 0.85
                    
                    print(f"\n[CALIBRATION COMPLETE]")
                    print(f"Baseline EAR: {self.baseline_ear:.3f}")
                    print(f"Blink Threshold: {self.ear_threshold:.3f}")
                    print(f"Blink detection is now active!\n")
                
                self.is_calibrated = True
        
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
            self.in_dip = False  # Reset dip tracking since it's a full blink
            self.current_blink_start = timestamp
            
        # Detect end of blink and evaluate micro-sleep
        elif self.previous_state == 'closed' and current_state == 'open':
            if self.current_blink_start is not None:
                duration = timestamp - self.current_blink_start
                if duration >= 0.5:  # Micro-sleep threshold
                    self.micro_sleeps.append({
                        'start': self.current_blink_start,
                        'end': timestamp,
                        'duration': duration
                    })
                self.current_blink_start = None
            
        # Detect partial blink
        if not blink_detected and self.is_calibrated:
            # We use the dynamic ear_threshold as the boundary for a full blink.
            # Start of a dip begins slightly above the ear_threshold
            dynamic_partial_start = self.ear_threshold + (self.baseline_ear - self.ear_threshold) * 0.4
            
            if avg_ear < dynamic_partial_start and not self.in_dip and current_state == 'open':
                self.in_dip = True
                self.current_dip_min_ear = avg_ear
            
            # Inside a dip, track minimum
            elif self.in_dip:
                if avg_ear < self.current_dip_min_ear:
                    self.current_dip_min_ear = avg_ear
                
                # Eye recovers from dip WITHOUT fully closing
                if avg_ear > dynamic_partial_start:
                    self.in_dip = False
                    # Register if it dropped somewhat significantly but wasn't a full blink
                    if self.current_dip_min_ear < (self.ear_threshold + (self.baseline_ear - self.ear_threshold) * 0.2):
                        self.partial_blinks.append({
                            'timestamp': timestamp - 0.2, # approximate center of dip
                            'min_ear': self.current_dip_min_ear
                        })
        
        self.previous_state = current_state
        
        return blink_detected
    
    def get_calibration_status(self):
        """
        Get current calibration status.
        
        Returns:
            Dictionary with calibration info
        """
        if self.is_calibrated:
            return {
                'is_calibrated': True,
                'baseline_ear': self.baseline_ear,
                'threshold': self.ear_threshold,
                'progress': 1.0
            }
        
        if self.calibration_start_time is None:
            return {
                'is_calibrated': False,
                'progress': 0.0,
                'message': 'Waiting to start calibration...'
            }
        
        elapsed = time.time() - self.calibration_start_time
        progress = min(1.0, elapsed / self.calibration_duration)
        remaining = max(0, self.calibration_duration - elapsed)
        
        return {
            'is_calibrated': False,
            'progress': progress,
            'remaining_seconds': remaining,
            'message': f'Calibrating... Keep eyes open! ({remaining:.1f}s remaining)'
        }
    
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
        Calculate an eye/blink dementia-risk score (0-100%).

        This is a heuristic *screening* score built only from signals that
        are actually measured during a live session, so the weighting is
        honest about what the system observes:

            Blink rate abnormality ......... 28 pts
            Blink-pattern irregularity ..... 22 pts
            Reduced eye openness (low EAR) . 18 pts
            Micro-sleeps (eyes closed >0.5s) 17 pts
            Partial / incomplete blinks .... 15 pts
                                             ------
                                             100 pts

        NOTE: Earlier versions reserved 25 points for a CNN "sleepy" state.
        That model is only run in the Flask web build, never in the GUI, so
        in the GUI those points could never be awarded and scores silently
        capped near 75. The CNN signal is now an *optional bonus* applied
        only when real predictions are present, and the core 100 points come
        from signals the GUI genuinely captures.

        Returns:
            dict with risk_score, risk_level, risk_factors and a
            score_breakdown of how each factor contributed.
        """
        risk_factors = []
        breakdown = {}

        # 1. Blink Rate Analysis (28 points) -----------------------------
        blink_rate = self.calculate_blink_rate()
        rate_pts = 0.0
        if blink_rate < self.NORMAL_BLINK_RATE_MIN:
            deviation = (self.NORMAL_BLINK_RATE_MIN - blink_rate) / self.NORMAL_BLINK_RATE_MIN
            rate_pts = min(28, deviation * 28)
            risk_factors.append(f"Low blink rate ({blink_rate:.1f} bpm)")
        elif blink_rate > self.NORMAL_BLINK_RATE_MAX:
            deviation = (blink_rate - self.NORMAL_BLINK_RATE_MAX) / self.NORMAL_BLINK_RATE_MAX
            rate_pts = min(28, deviation * 28)
            risk_factors.append(f"High blink rate ({blink_rate:.1f} bpm)")
        breakdown['blink_rate'] = round(rate_pts, 1)

        # 2. Blink Pattern Regularity (22 points) -------------------------
        variance = self.calculate_blink_variance()
        var_pts = 0.0
        if variance > 15.0:  # Threshold for truly erratic patterns (research-based)
            var_pts = min(22, (variance / 20.0) * 22)
            risk_factors.append(f"Highly erratic blink pattern (var: {variance:.2f})")
        breakdown['blink_variance'] = round(var_pts, 1)

        # 3. Low Eye Aspect Ratio (18 points) -----------------------------
        avg_ear = 0.0
        ear_pts = 0.0
        if self.ear_history:
            recent_ears = [e['ear'] for e in list(self.ear_history)[-100:]]
            avg_ear = float(np.mean(recent_ears))
            if avg_ear < self.NORMAL_EAR_MIN:
                deviation = (self.NORMAL_EAR_MIN - avg_ear) / self.NORMAL_EAR_MIN
                ear_pts = min(18, deviation * 18)
                risk_factors.append(f"Low eye openness (EAR: {avg_ear:.3f})")
        breakdown['eye_openness'] = round(ear_pts, 1)

        # 4. Micro-sleeps (17 points) -------------------------------------
        # Each prolonged eye closure (>0.5 s) is a strong attention/drowsiness
        # indicator. ~6 points each, capped.
        n_micro = len(self.micro_sleeps)
        micro_pts = min(17, n_micro * 6)
        if n_micro > 0:
            risk_factors.append(f"{n_micro} micro-sleep(s) detected (>0.5s closure)")
        breakdown['micro_sleeps'] = round(micro_pts, 1)

        # 5. Partial / incomplete blinks (15 points) ----------------------
        # Frequent incomplete blinks reflect reduced blink amplitude/control.
        n_partial = len(self.partial_blinks)
        partial_pts = min(15, n_partial * 2)
        if n_partial >= 3:
            risk_factors.append(f"{n_partial} partial/incomplete blinks")
        breakdown['partial_blinks'] = round(partial_pts, 1)

        risk_score = rate_pts + var_pts + ear_pts + micro_pts + partial_pts

        # Optional bonus: CNN sleepy-state, ONLY when real predictions exist
        # (Flask web build). Never penalises the GUI build where it's absent.
        avg_pred = self.get_avg_prediction(window_seconds=30)
        if self.predictions_history and avg_pred['sleepy'] > 0.6:
            bonus = (avg_pred['sleepy'] - 0.6) / 0.4 * 15
            risk_score += bonus
            breakdown['sleepy_state_bonus'] = round(bonus, 1)
            risk_factors.append(f"High sleepy state ({avg_pred['sleepy']*100:.1f}%)")

        # Cap at 100
        risk_score = min(100, risk_score)

        return {
            'risk_score': risk_score,
            'risk_level': self._get_risk_level(risk_score),
            'risk_factors': risk_factors,
            'blink_rate': blink_rate,
            'blink_variance': variance,
            'avg_ear': avg_ear,
            'avg_predictions': avg_pred,
            'total_blinks': self.blink_count,
            'score_breakdown': breakdown
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
            'avg_ear': np.mean([e['ear'] for e in self.ear_history]) if self.ear_history else 0.0,
            'partial_blinks': self.partial_blinks,
            'micro_sleeps': self.micro_sleeps
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
        
        self.micro_sleeps.clear()
        self.current_blink_start = None
        
        self.in_dip = False
        self.partial_blinks.clear()
        
        # Reset calibration
        self.is_calibrated = False
        self.calibration_samples = []
        self.calibration_start_time = None
        self.baseline_ear = None
        self.ear_threshold = 0.30  # Reset to default
    
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
