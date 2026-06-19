"""
Flask Web Application for Dementia Detection
Real-time webcam analysis with web dashboard.
"""

from flask import Flask, render_template, Response, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import json
import time

from webcam_detector import WebcamEyeDetector
from dementia_analyzer import DementiaAnalyzer

app = Flask(__name__)
CORS(app)

# Global variables
detector = None
analyzer = None
model = None
is_running = False

# Load trained model
MODEL_PATH = Path('models/eye_classifier_best.h5')

def load_model():
    """Load the trained CNN model."""
    global model
    
    if MODEL_PATH.exists():
        print(f"Loading model from {MODEL_PATH}...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully!")
        return True
    else:
        print(f"Model not found at {MODEL_PATH}")
        print("Please train the model first using train_model.py")
        return False

def initialize_system():
    """Initialize webcam detector and analyzer."""
    global detector, analyzer
    
    detector = WebcamEyeDetector(img_size=(64, 64))
    analyzer = DementiaAnalyzer()
    
    try:
        detector.start_camera()
        return True
    except Exception as e:
        print(f"Error starting camera: {e}")
        return False

def generate_frames():
    """Generate video frames with analysis overlay."""
    global detector, analyzer, model, is_running
    
    while is_running:
        # Capture frame
        frame = detector.capture_frame()
        
        if frame is None:
            continue
        
        # Process frame
        result = detector.process_frame(frame)
        
        if result['face_detected']:
            # Draw eye boxes
            annotated = detector.draw_eye_boxes(
                frame,
                result['left_eye_bbox'],
                result['right_eye_bbox']
            )
            
            # Get predictions if model is loaded
            if model is not None and result['left_eye_preprocessed'] is not None:
                # Predict on left eye
                prediction = model.predict(result['left_eye_preprocessed'], verbose=0)[0]
                pred_awake = float(prediction[0])
                pred_sleepy = float(prediction[1])
                
                # Analyze predictions
                analyzer.analyze_predictions(pred_awake, pred_sleepy)
                
                # Display prediction
                label = "AWAKE" if pred_awake > pred_sleepy else "SLEEPY"
                confidence = max(pred_awake, pred_sleepy) * 100
                color = (0, 255, 0) if label == "AWAKE" else (0, 165, 255)
                
                cv2.putText(annotated, f"{label} ({confidence:.1f}%)", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Detect blinks
            blink_detected = analyzer.detect_blink(
                result['left_ear'],
                result['right_ear']
            )
            
            # Display blink info
            if blink_detected:
                cv2.putText(annotated, "BLINK!", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Display EAR
            cv2.putText(annotated, f"EAR: {result['left_ear']:.3f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display blink rate
            blink_rate = analyzer.calculate_blink_rate()
            cv2.putText(annotated, f"Blink Rate: {blink_rate:.1f} bpm", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        else:
            annotated = frame
            cv2.putText(annotated, "No face detected", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', annotated)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start', methods=['POST'])
def start_detection():
    """Start detection."""
    global is_running, detector, analyzer
    
    if not is_running:
        if detector is None:
            if not initialize_system():
                return jsonify({'success': False, 'message': 'Failed to start camera'})
        
        is_running = True
        analyzer.reset()
        return jsonify({'success': True, 'message': 'Detection started'})
    
    return jsonify({'success': False, 'message': 'Already running'})

@app.route('/stop', methods=['POST'])
def stop_detection():
    """Stop detection."""
    global is_running
    
    is_running = False
    return jsonify({'success': True, 'message': 'Detection stopped'})

@app.route('/metrics')
def get_metrics():
    """Get current metrics."""
    global analyzer
    
    if analyzer is None:
        return jsonify({'error': 'Analyzer not initialized'})
    
    # Get risk assessment
    risk = analyzer.calculate_dementia_risk()
    
    # Get session stats
    stats = analyzer.get_session_stats()
    
    # Get recent EAR values for graph
    recent_ear = []
    if analyzer.ear_history:
        recent_ear = [
            {'time': e['timestamp'], 'ear': e['ear']} 
            for e in list(analyzer.ear_history)[-100:]
        ]
    
    # Get recent blinks for graph
    recent_blinks = []
    if analyzer.blink_frames:
        recent_blinks = list(analyzer.blink_frames)[-50:]
    
    # Get calibration status
    calibration_status = analyzer.get_calibration_status()
    
    return jsonify({
        'risk_score': risk['risk_score'],
        'risk_level': risk['risk_level'],
        'risk_factors': risk['risk_factors'],
        'blink_rate': risk['blink_rate'],
        'total_blinks': risk['total_blinks'],
        'session_duration': stats['session_duration'],
        'avg_ear': stats['avg_ear'],
        'ear_history': recent_ear,
        'blink_times': recent_blinks,
        'avg_predictions': risk['avg_predictions'],
        'calibration': calibration_status
    })

@app.route('/export', methods=['POST'])
def export_data():
    """Export session data."""
    global analyzer
    
    if analyzer is None:
        return jsonify({'success': False, 'message': 'No data to export'})
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filepath = f'session_data_{timestamp}.csv'
    
    analyzer.export_session_data(filepath)
    
    return jsonify({'success': True, 'filepath': filepath})

if __name__ == '__main__':
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║          Dementia Detection System - Web Application                ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Load model
    model_loaded = load_model()
    
    if not model_loaded:
        print("\n[WARNING] Model not loaded. Train the model first!")
        print("Run: python train_model.py")
        print("\nContinuing without model (blink detection only)...\n")
    
    # Run Flask app
    print("\nStarting web server...")
    print("Open your browser and go to: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server\n")
    
    app.run(debug=True, threaded=True, host='0.0.0.0', port=5000)
