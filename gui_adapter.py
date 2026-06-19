"""
GUI Adapter for Dementia Analyzer
Bridges the GUI application with the dementia analyzer and webcam detector
"""

import cv2
import numpy as np
import time
from datetime import datetime
from dementia_analyzer import DementiaAnalyzer as CoreDementiaAnalyzer
from mediapipe_detector import MediaPipeEyeDetector


class GUIDementiaAnalyzer:
    """Adapter class that integrates analyzer with GUI requirements"""
    
    def __init__(self):
        """Initialize the GUI adapter"""
        self.analyzer = CoreDementiaAnalyzer(window_size=60)
        
        # Use high-accuracy MediaPipe 468-point face mesh detector
        self.detector = MediaPipeEyeDetector()
        
        # Session tracking
        self.session_active = False
        self.frame_count = 0
        
    def start_session(self):
        """Start a new detection session"""
        self.analyzer.reset()
        self.session_active = True
        self.frame_count = 0
        
    def stop_session(self):
        """Stop the current session"""
        self.session_active = False
        
    def reset(self):
        """Reset the analyzer for a new session"""
        self.analyzer.reset()
        self.session_active = False
        self.frame_count = 0
        
    def process_frame(self, frame):
        """
        Process a single frame for dementia detection
        
        Args:
            frame: OpenCV BGR image
            
        Returns:
            Annotated frame with detection overlays
        """
        if not self.session_active:
            return frame
            
        self.frame_count += 1
        
        # Process frame using the original Haar cascade detector
        result = self.detector.process_frame(frame)
        annotated_frame = frame.copy()
        
        if result['face_detected']:
            # Draw eye landmarks (MediaPipe specific method)
            if 'face_landmarks' in result and result['face_landmarks']:
                annotated_frame = self.detector.draw_landmarks(
                    annotated_frame,
                    result['face_landmarks']
                )
            
            # Get EAR values computed by the detector
            left_ear = result['left_ear']
            right_ear = result['right_ear']
            avg_ear = (left_ear + right_ear) / 2.0
            
            # Detect blink using analyzer's logic (must pass timestamp)
            current_time = time.time()
            is_blink = self.analyzer.detect_blink(left_ear, right_ear, timestamp=current_time)
                
            # Display metrics on frame
            y_offset = 30
            cv2.putText(annotated_frame, f"EAR: {avg_ear:.3f}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            y_offset += 30
            blink_status = "BLINK!" if is_blink else "Open"
            color = (0, 0, 255) if is_blink else (0, 255, 0)
            cv2.putText(annotated_frame, f"Status: {blink_status}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            y_offset += 30
            blink_rate = self.analyzer.calculate_blink_rate()
            cv2.putText(annotated_frame, f"Blink Rate: {blink_rate:.1f} bpm", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                       
        else:
            # No face detected
            cv2.putText(annotated_frame, "No face detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                       
        # Add frame counter
        h, w, _ = frame.shape
        cv2.putText(annotated_frame, f"Frame: {self.frame_count}", (w - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                   
        return annotated_frame
        
    def get_metrics(self):
        """
        Get current detection metrics for GUI display
        
        Returns:
            Dictionary with all metrics
        """
        stats = self.analyzer.get_session_stats()
        risk_data = self.analyzer.calculate_dementia_risk()
        risk_score = risk_data['risk_score']
        risk_level = risk_data['risk_level']
        
        # Determine risk factors
        risk_factors = []
        blink_rate = stats['blink_rate']
        
        if blink_rate < 12:
            risk_factors.append(f"Low blink rate ({blink_rate:.1f} bpm)")
        elif blink_rate > 25:
            risk_factors.append(f"High blink rate ({blink_rate:.1f} bpm)")
            
        if stats['blink_variance'] > 5.0:
            risk_factors.append(f"Irregular blink pattern (variance: {stats['blink_variance']:.2f})")
            
        if stats['avg_ear'] < 0.2:
            risk_factors.append(f"Low eye openness (EAR: {stats['avg_ear']:.3f})")
            
        # Get calibration status
        calibration = self.analyzer.get_calibration_status()
        if not calibration['is_calibrated']:
            risk_factors.append(f"Calibrating... ({calibration['progress']:.0f}%)")
            
        return {
            'blink_rate': stats['blink_rate'],
            'total_blinks': stats['total_blinks'],
            'avg_ear': stats['avg_ear'],
            'session_duration': stats['session_duration'],
            'risk_score': risk_score,
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'ear_history': list(self.analyzer.ear_history),
            'blink_times': [b['timestamp'] for b in self.analyzer.blink_history],
            'partial_blinks': self.analyzer.partial_blinks,
            'micro_sleeps': self.analyzer.micro_sleeps,
            'current_threshold': self.analyzer.ear_threshold,
            'frame_count': self.frame_count,
            'score_breakdown': risk_data.get('score_breakdown', {})
        }
        
    def export_data(self):
        """
        Export session data to CSV file
        
        Returns:
            Filepath of exported data
        """
        if self.analyzer.blink_count == 0 and len(self.analyzer.ear_history) == 0:
            return None
            
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"session_data_{timestamp}.csv"
        
        # Export using analyzer's method
        self.analyzer.export_session_data(filepath)
        
        return filepath
        
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'detector'):
            self.detector.close()

