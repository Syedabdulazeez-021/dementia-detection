"""
MediaPipe Eye Detector
Utilizes MediaPipe Face Mesh for high-accuracy 468-point facial landmark tracking
to compute precise Eye Aspect Ratio (EAR) for blink detection.
"""

import cv2
import numpy as np
import mediapipe as mp

class MediaPipeEyeDetector:
    """Detects eye landmarks using MediaPipe Face Mesh."""
    
    def __init__(self):
        """Initialize the MediaPipe Face Mesh module."""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Landmark indices for eyes
        # Right eye (from user's perspective, left side of image)
        self.right_eye_indices = [362, 385, 387, 263, 373, 380]
        # Left eye (from user's perspective, right side of image)
        self.left_eye_indices = [33, 160, 158, 133, 153, 144]

    def _calculate_ear(self, eye_landmarks):
        """
        Calculate the Eye Aspect Ratio (EAR) given 6 eye landmarks.
        Uses the standard EAR formula.
        
        Args:
            eye_landmarks: List of (x, y) coordinates for the 6 points
                           [P1, P2, P3, P4, P5, P6]
        Returns:
            Computed EAR value.
        """
        # P2 to P6
        v1 = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
        # P3 to P5
        v2 = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
        # P1 to P4
        h = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
        
        if h == 0:
            return 0.0
            
        ear = (v1 + v2) / (2.0 * h)
        return ear

    def process_frame(self, frame):
        """
        Process a frame to extract precise eye landmarks and EAR.
        
        Args:
            frame: Input OpenCV BGR frame
            
        Returns:
            Dictionary containing detection status, EARs, and landmarks.
        """
        # Convert the BGR image to RGB before processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the image with MediaPipe
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return {
                'face_detected': False
            }
            
        h, w, _ = frame.shape
        face_landmarks = results.multi_face_landmarks[0]
        
        # Extract coordinates
        right_eye_coords = []
        for idx in self.right_eye_indices:
            landmark = face_landmarks.landmark[idx]
            right_eye_coords.append((int(landmark.x * w), int(landmark.y * h)))
            
        left_eye_coords = []
        for idx in self.left_eye_indices:
            landmark = face_landmarks.landmark[idx]
            left_eye_coords.append((int(landmark.x * w), int(landmark.y * h)))
            
        # Calculate EARs
        right_ear = self._calculate_ear(right_eye_coords)
        left_ear = self._calculate_ear(left_eye_coords)
        
        return {
            'face_detected': True,
            'left_ear': left_ear,
            'right_ear': right_ear,
            'face_landmarks': face_landmarks
        }
        
    def draw_landmarks(self, frame, face_landmarks):
        """
        Draw the eye landmarks on the frame for visualization.
        
        Args:
            frame: Input BGR frame
            face_landmarks: MediaPipe face landmarks object
            
        Returns:
            Annotated frame
        """
        h, w, _ = frame.shape
        annotated = frame.copy()
        
        # Draw right eye (teal)
        for idx in self.right_eye_indices:
            landmark = face_landmarks.landmark[idx]
            pt = (int(landmark.x * w), int(landmark.y * h))
            cv2.circle(annotated, pt, 2, (139, 139, 46), -1) # BGR for teal-ish
            
        # Draw left eye (teal)
        for idx in self.left_eye_indices:
            landmark = face_landmarks.landmark[idx]
            pt = (int(landmark.x * w), int(landmark.y * h))
            cv2.circle(annotated, pt, 2, (139, 139, 46), -1)
            
        return annotated

    def close(self):
        """Release MediaPipe resources."""
        self.face_mesh.close()
