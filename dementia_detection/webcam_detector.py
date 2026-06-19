"""
Webcam Eye Detector
Captures video from webcam, detects faces and eyes, extracts eye regions for analysis.
"""

import cv2
import numpy as np
from pathlib import Path
import mediapipe as mp

class WebcamEyeDetector:
    """Detects and extracts eye regions from webcam feed."""
    
    def __init__(self, img_size=(64, 64)):
        self.img_size = img_size
        self.cap = None
        
        # Initialize MediaPipe Face Mesh for eye detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Eye landmark indices (MediaPipe Face Mesh)
        self.LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
        
    def start_camera(self, camera_index=0):
        """Start webcam capture."""
        self.cap = cv2.VideoCapture(camera_index)
        
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam!")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Webcam started successfully!")
        return True
    
    def stop_camera(self):
        """Stop webcam capture."""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
    
    def get_eye_region(self, frame, eye_indices, padding=10):
        """
        Extract eye region from frame using landmarks.
        
        Args:
            frame: Input frame
            eye_indices: Landmark indices for the eye
            padding: Padding around eye region
            
        Returns:
            Eye region image or None
        """
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None, None
        
        # Get first face
        face_landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]
        
        # Get eye landmarks
        eye_points = []
        for idx in eye_indices:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            eye_points.append((x, y))
        
        # Get bounding box
        eye_points = np.array(eye_points)
        x_min = max(0, eye_points[:, 0].min() - padding)
        x_max = min(w, eye_points[:, 0].max() + padding)
        y_min = max(0, eye_points[:, 1].min() - padding)
        y_max = min(h, eye_points[:, 1].max() + padding)
        
        # Extract eye region
        eye_region = frame[y_min:y_max, x_min:x_max]
        
        return eye_region, (x_min, y_min, x_max, y_max)
    
    def preprocess_eye(self, eye_region):
        """
        Preprocess eye region for model input.
        
        Args:
            eye_region: Raw eye image
            
        Returns:
            Preprocessed image ready for CNN
        """
        if eye_region is None or eye_region.size == 0:
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        
        # Resize to model input size
        resized = cv2.resize(gray, self.img_size)
        
        # Normalize to 0-1
        normalized = resized / 255.0
        
        # Add batch and channel dimensions
        preprocessed = normalized.reshape(1, *self.img_size, 1)
        
        return preprocessed
    
    def calculate_eye_aspect_ratio(self, eye_region):
        """
        Calculate Eye Aspect Ratio (EAR) for blink detection.
        Lower EAR indicates closed eye.
        
        Args:
            eye_region: Eye image
            
        Returns:
            EAR value (float)
        """
        if eye_region is None or eye_region.size == 0:
            return 0.0
        
        # Simple EAR based on average brightness
        gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY) if len(eye_region.shape) == 3 else eye_region
        avg_brightness = np.mean(gray)
        
        # Normalize to 0-1 range
        ear = avg_brightness / 255.0
        
        return ear
    
    def capture_frame(self):
        """
        Capture a single frame from webcam.
        
        Returns:
            Frame or None if capture failed
        """
        if self.cap is None or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        
        if not ret:
            return None
        
        return frame
    
    def process_frame(self, frame):
        """
        Process frame to extract both eyes.
        
        Args:
            frame: Input frame
            
        Returns:
            Dictionary with eye data
        """
        result = {
            'frame': frame,
            'left_eye': None,
            'right_eye': None,
            'left_eye_bbox': None,
            'right_eye_bbox': None,
            'left_eye_preprocessed': None,
            'right_eye_preprocessed': None,
            'left_ear': 0.0,
            'right_ear': 0.0,
            'face_detected': False
        }
        
        # Extract left eye
        left_eye, left_bbox = self.get_eye_region(frame, self.LEFT_EYE_INDICES)
        if left_eye is not None:
            result['left_eye'] = left_eye
            result['left_eye_bbox'] = left_bbox
            result['left_eye_preprocessed'] = self.preprocess_eye(left_eye)
            result['left_ear'] = self.calculate_eye_aspect_ratio(left_eye)
            result['face_detected'] = True
        
        # Extract right eye
        right_eye, right_bbox = self.get_eye_region(frame, self.RIGHT_EYE_INDICES)
        if right_eye is not None:
            result['right_eye'] = right_eye
            result['right_eye_bbox'] = right_bbox
            result['right_eye_preprocessed'] = self.preprocess_eye(right_eye)
            result['right_ear'] = self.calculate_eye_aspect_ratio(right_eye)
        
        return result
    
    def draw_eye_boxes(self, frame, left_bbox, right_bbox):
        """Draw bounding boxes around detected eyes."""
        annotated = frame.copy()
        
        if left_bbox is not None:
            x1, y1, x2, y2 = left_bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, "Left Eye", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if right_bbox is not None:
            x1, y1, x2, y2 = right_bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, "Right Eye", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return annotated


def test_webcam_detector():
    """Test the webcam eye detector."""
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║              Webcam Eye Detector - Test Mode                         ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    detector = WebcamEyeDetector()
    
    try:
        # Start camera
        print("\nStarting webcam...")
        detector.start_camera()
        
        print("\nWebcam is running!")
        print("Press 'q' to quit")
        print("Press 's' to save current eye images")
        
        save_counter = 0
        
        while True:
            # Capture frame
            frame = detector.capture_frame()
            
            if frame is None:
                print("Failed to capture frame!")
                break
            
            # Process frame
            result = detector.process_frame(frame)
            
            # Draw eye boxes
            if result['face_detected']:
                annotated = detector.draw_eye_boxes(
                    frame, 
                    result['left_eye_bbox'], 
                    result['right_eye_bbox']
                )
                
                # Display EAR values
                cv2.putText(annotated, f"Left EAR: {result['left_ear']:.3f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated, f"Right EAR: {result['right_ear']:.3f}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                annotated = frame
                cv2.putText(annotated, "No face detected", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display frame
            cv2.imshow('Webcam Eye Detector', annotated)
            
            # Display individual eyes if detected
            if result['left_eye'] is not None:
                cv2.imshow('Left Eye', result['left_eye'])
            if result['right_eye'] is not None:
                cv2.imshow('Right Eye', result['right_eye'])
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s') and result['face_detected']:
                # Save eye images
                if result['left_eye'] is not None:
                    cv2.imwrite(f'left_eye_{save_counter}.png', result['left_eye'])
                if result['right_eye'] is not None:
                    cv2.imwrite(f'right_eye_{save_counter}.png', result['right_eye'])
                print(f"Saved eye images: {save_counter}")
                save_counter += 1
        
    finally:
        detector.stop_camera()
        print("\nWebcam stopped.")


if __name__ == "__main__":
    test_webcam_detector()
