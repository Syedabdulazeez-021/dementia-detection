"""
Webcam Eye Detector (OpenCV Version)
Captures video from webcam, detects faces and eyes using Haar Cascades.
This version uses OpenCV only - no MediaPipe dependency.
"""

import cv2
import numpy as np
from pathlib import Path

class WebcamEyeDetector:
    """Detects and extracts eye regions from webcam feed using OpenCV."""
    
    def __init__(self, img_size=(64, 64)):
        self.img_size = img_size
        self.cap = None
        
        # Load Haar Cascade classifiers
        cascade_path = cv2.data.haarcascades
        self.face_cascade = cv2.CascadeClassifier(cascade_path + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cascade_path + 'haarcascade_eye.xml')
        
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
    
    def get_eye_region(self, frame):
        """
        Extract eye regions from frame using Haar Cascades.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (left_eye, right_eye, bboxes)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return None, None, None, None
        
        # Get first face
        (x, y, w, h) = faces[0]
        face_roi = gray[y:y+h, x:x+w]
        face_roi_color = frame[y:y+h, x:x+w]
        
        # Detect eyes in face region
        eyes = self.eye_cascade.detectMultiScale(face_roi, 1.1, 5)
        
        if len(eyes) < 2:
            return None, None, None, None
        
        # Sort eyes by x-coordinate (left to right)
        eyes = sorted(eyes, key=lambda e: e[0])
        
        # Extract left and right eyes
        left_eye_bbox = eyes[0]
        right_eye_bbox = eyes[1] if len(eyes) > 1 else eyes[0]
        
        # Get eye regions
        (ex1, ey1, ew1, eh1) = left_eye_bbox
        left_eye = face_roi_color[ey1:ey1+eh1, ex1:ex1+ew1]
        
        (ex2, ey2, ew2, eh2) = right_eye_bbox
        right_eye = face_roi_color[ey2:ey2+eh2, ex2:ex2+ew2]
        
        # Convert to absolute coordinates
        left_bbox = (x + ex1, y + ey1, x + ex1 + ew1, y + ey1 + eh1)
        right_bbox = (x + ex2, y + ey2, x + ex2 + ew2, y + ey2 + eh2)
        
        return left_eye, right_eye, left_bbox, right_bbox
    
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
        if len(eye_region.shape) == 3:
            gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = eye_region
        
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
        Analyzes pixel intensity to detect eyelid closure.
        Lower EAR indicates closed eye.
        
        Args:
            eye_region: Eye image
            
        Returns:
            EAR value (float)
        """
        if eye_region is None or eye_region.size == 0:
            return 0.0
        
        # Convert to grayscale if needed
        if len(eye_region.shape) == 3:
            gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = eye_region
        
        height, width = gray.shape
        
        if height < 10 or width < 10:
            return 0.0
        
        # Method 1: Vertical intensity profile
        # When eye is open: dark pupil in center, bright sclera top/bottom
        # When eye is closed: uniform dark eyelid across entire region
        
        # Calculate mean intensity for each row
        row_means = np.mean(gray, axis=1)
        
        # Find the darkest region (pupil when open, eyelid when closed)
        min_intensity = np.min(row_means)
        max_intensity = np.max(row_means)
        intensity_range = max_intensity - min_intensity
        
        # Method 2: Edge detection
        # Open eye has strong horizontal edges (eyelid lines)
        # Closed eye has fewer edges
        edges = cv2.Canny(gray, 30, 100)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Method 3: Variance in vertical direction
        # Open eye has high variance (dark pupil, bright sclera)
        # Closed eye has low variance (uniform eyelid)
        vertical_variance = np.var(row_means)
        
        # Combine metrics into EAR
        # Normalize each component
        intensity_score = intensity_range / 255.0  # 0-1
        edge_score = min(edge_density * 10, 1.0)   # 0-1
        variance_score = min(vertical_variance / 1000.0, 1.0)  # 0-1
        
        # Weighted combination
        ear = (intensity_score * 0.4 + edge_score * 0.3 + variance_score * 0.3)
        
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
        
        # Extract eyes
        left_eye, right_eye, left_bbox, right_bbox = self.get_eye_region(frame)
        
        if left_eye is not None and right_eye is not None:
            result['left_eye'] = left_eye
            result['right_eye'] = right_eye
            result['left_eye_bbox'] = left_bbox
            result['right_eye_bbox'] = right_bbox
            result['left_eye_preprocessed'] = self.preprocess_eye(left_eye)
            result['right_eye_preprocessed'] = self.preprocess_eye(right_eye)
            result['left_ear'] = self.calculate_eye_aspect_ratio(left_eye)
            result['right_ear'] = self.calculate_eye_aspect_ratio(right_eye)
            result['face_detected'] = True
        
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
