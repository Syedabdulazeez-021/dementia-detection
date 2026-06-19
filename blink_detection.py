"""
Blink Rate Detection and Analysis Module
This module analyzes eye images to detect blinks and calculate blink rates.
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
import json


class BlinkDetector:
    """Class for detecting blinks in eye images."""
    
    def __init__(self, eye_aspect_ratio_threshold=0.25):
        """
        Initialize blink detector.
        
        Args:
            eye_aspect_ratio_threshold: Threshold for detecting closed eyes
        """
        self.ear_threshold = eye_aspect_ratio_threshold
        self.blink_history = []
    
    def calculate_eye_aspect_ratio(self, eye_image):
        """
        Calculate Eye Aspect Ratio (EAR) from an eye image.
        Lower EAR indicates closed eye (blink).
        
        Args:
            eye_image: Grayscale eye image
            
        Returns:
            EAR value (float)
        """
        # Calculate average pixel intensity
        # Darker images (closed eyes) have lower intensity
        avg_intensity = np.mean(eye_image)
        
        # Normalize to 0-1 range
        normalized_intensity = avg_intensity / 255.0
        
        return normalized_intensity
    
    def is_eye_closed(self, eye_image):
        """
        Determine if eye is closed based on image analysis.
        
        Args:
            eye_image: Grayscale eye image
            
        Returns:
            Boolean indicating if eye is closed
        """
        ear = self.calculate_eye_aspect_ratio(eye_image)
        return ear < self.ear_threshold
    
    def detect_blink_from_image(self, image_path):
        """
        Detect if an image shows a closed eye (blink).
        
        Args:
            image_path: Path to eye image
            
        Returns:
            Dictionary with blink detection results
        """
        # Load image
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Calculate metrics
        ear = self.calculate_eye_aspect_ratio(img)
        is_closed = self.is_eye_closed(img)
        
        # Additional metrics
        avg_intensity = np.mean(img)
        std_intensity = np.std(img)
        
        result = {
            'image_path': str(image_path),
            'eye_aspect_ratio': ear,
            'is_closed': is_closed,
            'avg_intensity': avg_intensity,
            'std_intensity': std_intensity,
            'classification': 'CLOSED' if is_closed else 'OPEN'
        }
        
        return result
    
    def analyze_image_sequence(self, image_paths, fps=30):
        """
        Analyze a sequence of images to detect blinks and calculate blink rate.
        
        Args:
            image_paths: List of image paths in temporal order
            fps: Frames per second (for time calculation)
            
        Returns:
            Dictionary with blink analysis results
        """
        results = []
        blink_count = 0
        previous_state = None
        blink_frames = []
        
        for idx, img_path in enumerate(image_paths):
            detection = self.detect_blink_from_image(img_path)
            detection['frame_number'] = idx
            detection['timestamp'] = idx / fps
            
            # Detect blink transitions (open -> closed)
            if previous_state is not None:
                if not previous_state and detection['is_closed']:
                    blink_count += 1
                    blink_frames.append(idx)
            
            previous_state = detection['is_closed']
            results.append(detection)
        
        # Calculate statistics
        total_time = len(image_paths) / fps
        blink_rate = (blink_count / total_time) * 60 if total_time > 0 else 0  # Blinks per minute
        
        closed_frames = sum(1 for r in results if r['is_closed'])
        open_frames = len(results) - closed_frames
        
        analysis = {
            'total_frames': len(image_paths),
            'total_time_seconds': total_time,
            'blink_count': blink_count,
            'blink_rate_per_minute': blink_rate,
            'closed_frames': closed_frames,
            'open_frames': open_frames,
            'closed_percentage': (closed_frames / len(results)) * 100 if results else 0,
            'blink_frames': blink_frames,
            'frame_results': results
        }
        
        return analysis


class DatasetAnalyzer:
    """Analyze the entire eye tracking dataset."""
    
    def __init__(self, dataset_path):
        """
        Initialize dataset analyzer.
        
        Args:
            dataset_path: Path to dataset root directory
        """
        self.dataset_path = Path(dataset_path)
        self.detector = BlinkDetector()
    
    def analyze_dataset_structure(self):
        """
        Analyze the structure and statistics of the dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'train': {'awake': 0, 'sleepy': 0},
            'val': {'awake': 0, 'sleepy': 0},
            'test': {'awake': 0, 'sleepy': 0}
        }
        
        for split in ['train', 'val', 'test']:
            for category in ['awake', 'sleepy']:
                path = self.dataset_path / split / category
                if path.exists():
                    images = list(path.glob('*.png')) + list(path.glob('*.jpg'))
                    stats[split][category] = len(images)
        
        return stats
    
    def sample_images_from_category(self, split='train', category='awake', num_samples=10):
        """
        Get random sample images from a category.
        
        Args:
            split: Dataset split ('train', 'val', 'test')
            category: Category ('awake', 'sleepy')
            num_samples: Number of samples to retrieve
            
        Returns:
            List of image paths
        """
        path = self.dataset_path / split / category
        if not path.exists():
            return []
        
        images = list(path.glob('*.png')) + list(path.glob('*.jpg'))
        
        if len(images) <= num_samples:
            return images
        
        # Random sampling
        indices = np.random.choice(len(images), num_samples, replace=False)
        return [images[i] for i in indices]
    
    def compare_categories(self, num_samples=50):
        """
        Compare awake vs sleepy categories using pixel analysis.
        
        Args:
            num_samples: Number of samples to analyze per category
            
        Returns:
            Comparison statistics
        """
        awake_samples = self.sample_images_from_category('train', 'awake', num_samples)
        sleepy_samples = self.sample_images_from_category('train', 'sleepy', num_samples)
        
        awake_results = []
        sleepy_results = []
        
        print(f"Analyzing {len(awake_samples)} awake images...")
        for img_path in awake_samples:
            result = self.detector.detect_blink_from_image(img_path)
            awake_results.append(result)
        
        print(f"Analyzing {len(sleepy_samples)} sleepy images...")
        for img_path in sleepy_samples:
            result = self.detector.detect_blink_from_image(img_path)
            sleepy_results.append(result)
        
        # Calculate statistics
        awake_stats = {
            'avg_ear': np.mean([r['eye_aspect_ratio'] for r in awake_results]),
            'avg_intensity': np.mean([r['avg_intensity'] for r in awake_results]),
            'closed_percentage': sum(1 for r in awake_results if r['is_closed']) / len(awake_results) * 100
        }
        
        sleepy_stats = {
            'avg_ear': np.mean([r['eye_aspect_ratio'] for r in sleepy_results]),
            'avg_intensity': np.mean([r['avg_intensity'] for r in sleepy_results]),
            'closed_percentage': sum(1 for r in sleepy_results if r['is_closed']) / len(sleepy_results) * 100
        }
        
        comparison = {
            'awake': awake_stats,
            'sleepy': sleepy_stats,
            'difference': {
                'ear_diff': awake_stats['avg_ear'] - sleepy_stats['avg_ear'],
                'intensity_diff': awake_stats['avg_intensity'] - sleepy_stats['avg_intensity'],
                'closed_percentage_diff': awake_stats['closed_percentage'] - sleepy_stats['closed_percentage']
            }
        }
        
        return comparison
    
    def visualize_category_comparison(self, num_samples=5):
        """
        Visualize sample images from both categories.
        
        Args:
            num_samples: Number of samples to display per category
        """
        awake_samples = self.sample_images_from_category('train', 'awake', num_samples)
        sleepy_samples = self.sample_images_from_category('train', 'sleepy', num_samples)
        
        fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
        
        for i in range(num_samples):
            # Awake images
            if i < len(awake_samples):
                img = cv2.imread(str(awake_samples[i]), cv2.IMREAD_GRAYSCALE)
                axes[0, i].imshow(img, cmap='gray')
                axes[0, i].set_title(f'Awake {i+1}')
                axes[0, i].axis('off')
            
            # Sleepy images
            if i < len(sleepy_samples):
                img = cv2.imread(str(sleepy_samples[i]), cv2.IMREAD_GRAYSCALE)
                axes[1, i].imshow(img, cmap='gray')
                axes[1, i].set_title(f'Sleepy {i+1}')
                axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.dataset_path / 'category_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()


def analyze_blink_rate(image_folder, fps=30, visualize=True):
    """
    Analyze blink rate from a folder of sequential images.
    
    Args:
        image_folder: Path to folder containing sequential eye images
        fps: Frames per second
        visualize: Whether to create visualizations
        
    Returns:
        Analysis results dictionary
    """
    detector = BlinkDetector()
    
    # Get all images sorted by name
    image_paths = sorted(Path(image_folder).glob('*.png')) + sorted(Path(image_folder).glob('*.jpg'))
    
    if not image_paths:
        print(f"No images found in {image_folder}")
        return None
    
    print(f"Analyzing {len(image_paths)} images...")
    analysis = detector.analyze_image_sequence(image_paths, fps)
    
    print(f"\n{'='*60}")
    print(f"BLINK RATE ANALYSIS RESULTS")
    print(f"{'='*60}")
    print(f"Total Frames:           {analysis['total_frames']}")
    print(f"Total Time:             {analysis['total_time_seconds']:.2f} seconds")
    print(f"Blinks Detected:        {analysis['blink_count']}")
    print(f"Blink Rate:             {analysis['blink_rate_per_minute']:.2f} blinks/minute")
    print(f"Open Frames:            {analysis['open_frames']} ({100 - analysis['closed_percentage']:.1f}%)")
    print(f"Closed Frames:          {analysis['closed_frames']} ({analysis['closed_percentage']:.1f}%)")
    print(f"{'='*60}\n")
    
    if visualize and len(image_paths) > 0:
        _visualize_blink_analysis(analysis)
    
    return analysis


def _visualize_blink_analysis(analysis):
    """Create visualizations for blink analysis."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Eye state over time
    frames = [r['frame_number'] for r in analysis['frame_results']]
    states = [1 if r['is_closed'] else 0 for r in analysis['frame_results']]
    
    axes[0].plot(frames, states, linewidth=2)
    axes[0].set_xlabel('Frame Number')
    axes[0].set_ylabel('Eye State')
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(['Open', 'Closed'])
    axes[0].set_title('Eye State Over Time')
    axes[0].grid(True, alpha=0.3)
    
    # Mark blinks
    for blink_frame in analysis['blink_frames']:
        axes[0].axvline(x=blink_frame, color='red', alpha=0.5, linestyle='--')
    
    # Plot 2: Eye Aspect Ratio over time
    ear_values = [r['eye_aspect_ratio'] for r in analysis['frame_results']]
    
    axes[1].plot(frames, ear_values, linewidth=2, color='green')
    axes[1].set_xlabel('Frame Number')
    axes[1].set_ylabel('Eye Aspect Ratio')
    axes[1].set_title('Eye Aspect Ratio Over Time')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0.25, color='red', linestyle='--', label='Blink Threshold')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Blink Rate Detection and Analysis Tool")
    print("=" * 60)
    print("\nThis module provides tools to detect blinks and analyze blink rates.")
    print("\nUsage examples:")
    print("  from blink_detection import analyze_blink_rate, DatasetAnalyzer")
    print("  analyze_blink_rate('path/to/image/folder')")
    print("  analyzer = DatasetAnalyzer('path/to/dataset')")
