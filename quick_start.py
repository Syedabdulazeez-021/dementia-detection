"""
Quick Start Script
Automated setup and launch for the Dementia Detection System
"""

import subprocess
import sys
import os
from pathlib import Path

def print_header(text):
    """Print formatted header."""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def check_python_version():
    """Check if Python version is compatible."""
    print_header("Checking Python Version")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        return False
    
    print("✅ Python version is compatible")
    return True

def install_dependencies():
    """Install required packages."""
    print_header("Installing Dependencies")
    
    print("Installing packages from requirements.txt...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("\n✅ All dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("\n❌ Failed to install dependencies!")
        return False

def check_dataset():
    """Check if dataset is available."""
    print_header("Checking Dataset")
    
    dataset_path = Path("blink rate")
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found at: {dataset_path}")
        print("\nPlease ensure the dataset is in the correct location:")
        print("  blink rate/")
        return False
    
    # Check for train/val/test folders
    train_path = dataset_path / "train"
    val_path = dataset_path / "val"
    test_path = dataset_path / "test"
    
    if not all([train_path.exists(), val_path.exists(), test_path.exists()]):
        print("❌ Dataset structure incomplete!")
        print("Required folders: train, val, test")
        return False
    
    print(f"✅ Dataset found at: {dataset_path}")
    return True

def check_model():
    """Check if trained model exists."""
    print_header("Checking Trained Model")
    
    model_path = Path("models/eye_classifier_best.h5")
    
    if not model_path.exists():
        print("⚠️  Trained model not found!")
        print("\nYou need to train the model first.")
        
        response = input("\nDo you want to train the model now? (y/n): ")
        
        if response.lower() == 'y':
            return train_model()
        else:
            print("\nYou can train later by running: python train_model.py")
            print("The system will run without ML predictions (blink detection only)")
            return True
    
    print(f"✅ Trained model found: {model_path}")
    return True

def train_model():
    """Train the CNN model."""
    print_header("Training CNN Model")
    
    print("This will take approximately 30 minutes...")
    print("Press Ctrl+C to cancel\n")
    
    try:
        subprocess.check_call([sys.executable, "train_model.py"])
        print("\n✅ Model training completed!")
        return True
    except subprocess.CalledProcessError:
        print("\n❌ Model training failed!")
        return False
    except KeyboardInterrupt:
        print("\n⚠️  Training cancelled by user")
        return False

def test_webcam():
    """Test webcam functionality."""
    print_header("Testing Webcam")
    
    try:
        import cv2
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Could not open webcam!")
            print("\nTroubleshooting:")
            print("  1. Check if webcam is connected")
            print("  2. Close other applications using the webcam")
            print("  3. Try a different camera index in the code")
            return False
        
        cap.release()
        print("✅ Webcam is working!")
        return True
        
    except Exception as e:
        print(f"❌ Webcam test failed: {e}")
        return False

def launch_app():
    """Launch the web application."""
    print_header("Launching Web Application")
    
    print("Starting Flask server...")
    print("\nThe application will open at: http://localhost:5000")
    print("Press Ctrl+C to stop the server\n")
    
    try:
        subprocess.check_call([sys.executable, "app.py"])
    except KeyboardInterrupt:
        print("\n\n✅ Server stopped by user")
    except subprocess.CalledProcessError:
        print("\n❌ Failed to start server!")

def main():
    """Main setup and launch sequence."""
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                                                                      ║
    ║          🧠 DEMENTIA DETECTION SYSTEM - QUICK START 🧠              ║
    ║                                                                      ║
    ║              Webcam-based Eye Tracking & Analysis                   ║
    ║                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Step 1: Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Step 2: Install dependencies
    response = input("Install/update dependencies? (y/n): ")
    if response.lower() == 'y':
        if not install_dependencies():
            sys.exit(1)
    
    # Step 3: Check dataset
    if not check_dataset():
        print("\n⚠️  Dataset not found. Some features may not work.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Step 4: Check/train model
    if not check_model():
        sys.exit(1)
    
    # Step 5: Test webcam
    if not test_webcam():
        response = input("\nWebcam test failed. Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Step 6: Launch application
    print_header("Setup Complete!")
    print("All checks passed! Ready to launch the application.\n")
    
    response = input("Launch web application now? (y/n): ")
    if response.lower() == 'y':
        launch_app()
    else:
        print("\nYou can launch the app later by running: python app.py")
        print("Or run this script again: python quick_start.py")
    
    print("\n" + "="*70)
    print("  Thank you for using the Dementia Detection System!")
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup cancelled by user")
        sys.exit(0)
