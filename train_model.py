"""
CNN Model Training for Eye Classification
Trains a Convolutional Neural Network on 84,898 eye images to classify awake vs sleepy states.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Configuration
DATASET_PATH = Path("blink rate")  # Relative path to dataset
IMG_SIZE = (64, 64)  # Resize images to 64x64
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

class EyeClassifierCNN:
    """CNN Model for classifying eye images as awake or sleepy."""
    
    def __init__(self, img_size=(64, 64)):
        self.img_size = img_size
        self.model = None
        self.history = None
        
    def build_model(self):
        """
        Build CNN architecture.
        
        Architecture:
        - 3 Convolutional blocks (Conv2D + MaxPooling + Dropout)
        - Flatten layer
        - Dense layers with dropout
        - Output: 2 classes (awake, sleepy)
        """
        model = models.Sequential([
            # Input layer
            layers.Input(shape=(*self.img_size, 1)),  # Grayscale images
            
            # Block 1: Learn basic edges and patterns
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 2: Learn more complex shapes
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 3: Learn high-level features
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Flatten and classify
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            
            # Output layer: 2 classes (awake, sleepy)
            layers.Dense(2, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        self.model = model
        return model
    
    def prepare_data(self):
        """
        Prepare data generators for training, validation, and testing.
        Uses data augmentation for better generalization.
        """
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,  # Normalize pixel values to 0-1
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Only rescaling for validation and test
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Load training data
        train_generator = train_datagen.flow_from_directory(
            DATASET_PATH / 'train',
            target_size=self.img_size,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            color_mode='grayscale',
            shuffle=True
        )
        
        # Load validation data
        val_generator = val_test_datagen.flow_from_directory(
            DATASET_PATH / 'val',
            target_size=self.img_size,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            color_mode='grayscale',
            shuffle=False
        )
        
        # Load test data
        test_generator = val_test_datagen.flow_from_directory(
            DATASET_PATH / 'test',
            target_size=self.img_size,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            color_mode='grayscale',
            shuffle=False
        )
        
        return train_generator, val_generator, test_generator
    
    def train(self, train_gen, val_gen, epochs=EPOCHS):
        """
        Train the model with callbacks for monitoring and optimization.
        """
        # Callbacks
        callbacks = [
            # Save best model
            ModelCheckpoint(
                'models/eye_classifier_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            
            # Early stopping if no improvement
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train model
        print("\n" + "="*70)
        print("TRAINING CNN MODEL")
        print("="*70)
        print(f"Training samples: {train_gen.samples}")
        print(f"Validation samples: {val_gen.samples}")
        print(f"Batch size: {BATCH_SIZE}")
        print(f"Epochs: {epochs}")
        print("="*70 + "\n")
        
        history = self.model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history = history
        return history
    
    def evaluate(self, test_gen):
        """Evaluate model on test set."""
        print("\n" + "="*70)
        print("EVALUATING MODEL ON TEST SET")
        print("="*70)
        
        results = self.model.evaluate(test_gen, verbose=1)
        
        print("\nTest Results:")
        print(f"  Loss: {results[0]:.4f}")
        print(f"  Accuracy: {results[1]*100:.2f}%")
        print(f"  Precision: {results[2]*100:.2f}%")
        print(f"  Recall: {results[3]*100:.2f}%")
        print("="*70 + "\n")
        
        return results
    
    def plot_training_history(self):
        """Plot training and validation metrics."""
        if self.history is None:
            print("No training history available!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Train')
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Train')
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('models/training_history.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("Training history plot saved to: models/training_history.png")
    
    def save_model(self, filepath='models/eye_classifier_final.h5'):
        """Save the trained model."""
        self.model.save(filepath)
        print(f"\nModel saved to: {filepath}")
        
        # Save model summary
        with open('models/model_summary.txt', 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))
        print(f"Model summary saved to: models/model_summary.txt")


def main():
    """Main training pipeline."""
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║          CNN Model Training for Eye Classification                   ║
    ║              Awake vs Sleepy Detection                               ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize model
    classifier = EyeClassifierCNN(img_size=IMG_SIZE)
    
    # Build architecture
    print("\nBuilding CNN architecture...")
    model = classifier.build_model()
    
    # Print model summary
    print("\nModel Architecture:")
    print("="*70)
    model.summary()
    print("="*70)
    
    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")
    
    # Prepare data
    print("\nPreparing data generators...")
    train_gen, val_gen, test_gen = classifier.prepare_data()
    
    print(f"\nClass indices: {train_gen.class_indices}")
    
    # Train model
    input("\nPress Enter to start training...")
    history = classifier.train(train_gen, val_gen, epochs=EPOCHS)
    
    # Evaluate on test set
    test_results = classifier.evaluate(test_gen)
    
    # Plot training history
    classifier.plot_training_history()
    
    # Save model
    classifier.save_model()
    
    # Save training metrics
    metrics = {
        'test_loss': float(test_results[0]),
        'test_accuracy': float(test_results[1]),
        'test_precision': float(test_results[2]),
        'test_recall': float(test_results[3]),
        'epochs_trained': len(history.history['loss']),
        'img_size': IMG_SIZE,
        'batch_size': BATCH_SIZE
    }
    
    with open('models/training_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print("\nFiles created:")
    print("  - models/eye_classifier_best.h5 (best model during training)")
    print("  - models/eye_classifier_final.h5 (final model)")
    print("  - models/training_history.png (training graphs)")
    print("  - models/model_summary.txt (model architecture)")
    print("  - models/training_metrics.json (performance metrics)")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
