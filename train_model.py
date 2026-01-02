"""
Training script for Plant Disease Classification using ResNet50
This script trains a ResNet50 model on plant disease images

Dataset: PlantVillage Dataset
Model Architecture: ResNet50 with Transfer Learning
Classes: 39 plant disease classes
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from PIL import Image
import os
import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

# =============================================================================
# CONFIGURATION: Dataset and Model Paths
# =============================================================================
# IMPORTANT: The dataset must be organized in the following structure:
#   dataset/
#   ├── raw/              # Original PlantVillage dataset (all classes as folders)
#   ├── train/            # Training split (auto-created if missing)
#   ├── val/              # Validation split (auto-created if missing)
#   └── test/             # Test split (auto-created if missing)
#
# If train/val/test folders don't exist, they will be created automatically
# from the raw/ folder using a 70/15/15 split.

# Get the project root directory (parent of App directory)
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent

# Dataset paths - relative to project root
DATASET_ROOT = PROJECT_ROOT / 'dataset'
RAW_DATA_PATH = DATASET_ROOT / 'raw'          # Source: PlantVillage dataset
TRAIN_DATA_PATH = DATASET_ROOT / 'train'      # Training split (70%)
VAL_DATA_PATH = DATASET_ROOT / 'val'          # Validation split (15%)
TEST_DATA_PATH = DATASET_ROOT / 'test'        # Test split (15%)

# Model and results paths
MODEL_SAVE_PATH = SCRIPT_DIR / 'trained_model.pth'
RESULTS_DIR = SCRIPT_DIR / 'training_results'
RESULTS_DIR.mkdir(exist_ok=True)

# =============================================================================
# HYPERPARAMETERS
# =============================================================================
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
NUM_CLASSES = 39  # Number of disease classes

# Train/val/test split ratios (must sum to 1.0)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# =============================================================================
# DATA TRANSFORMATIONS
# =============================================================================
# Training: Includes data augmentation to improve generalization
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

# Validation and Test: Only basic preprocessing (no augmentation)
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =============================================================================
# DEVICE CONFIGURATION
# =============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# =============================================================================
# DATASET PREPARATION
# =============================================================================
def prepare_dataset_if_needed():
    """
    Checks if train/val/test splits exist.
    If not, creates them from raw/ dataset using prepare_dataset.py functionality.
    """
    if TRAIN_DATA_PATH.exists() and VAL_DATA_PATH.exists() and TEST_DATA_PATH.exists():
        print("\n✓ Train/val/test splits already exist. Skipping dataset preparation.")
        return
    
    print("\n" + "="*70)
    print("DATASET PREPARATION: Creating train/val/test splits from raw/")
    print("="*70)
    
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"\n❌ ERROR: Raw dataset not found at: {RAW_DATA_PATH}\n"
            f"Please ensure the PlantVillage dataset is extracted to:\n"
            f"  {RAW_DATA_PATH}\n"
            f"\nThe raw/ folder should contain class folders like:\n"
            f"  dataset/raw/Apple___Apple_scab/\n"
            f"  dataset/raw/Apple___Black_rot/\n"
            f"  ... (etc.)\n"
        )
    
    # Import dataset preparation function
    sys.path.insert(0, str(SCRIPT_DIR))
    from prepare_dataset import organize_dataset
    
    print(f"Source: {RAW_DATA_PATH}")
    print(f"Output: {DATASET_ROOT}")
    print(f"Split ratios: Train={TRAIN_RATIO*100}%, Val={VAL_RATIO*100}%, Test={TEST_RATIO*100}%")
    print("\nOrganizing dataset (this may take a few minutes)...")
    
    organize_dataset(
        source_dir=str(RAW_DATA_PATH),
        output_dir=str(DATASET_ROOT),
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO
    )
    
    print("\n✓ Dataset preparation complete!")

# =============================================================================
# DATA LOADING
# =============================================================================
def load_data():
    """
    Load training, validation, and test datasets.
    
    Returns:
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
        test_loader: DataLoader for test set
        class_names: List of class names
    """
    print("\n" + "="*70)
    print("LOADING DATASETS")
    print("="*70)
    
    # Load datasets
    train_dataset = ImageFolder(root=str(TRAIN_DATA_PATH), transform=train_transform)
    val_dataset = ImageFolder(root=str(VAL_DATA_PATH), transform=val_test_transform)
    test_dataset = ImageFolder(root=str(TEST_DATA_PATH), transform=val_test_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Print dataset statistics
    print(f"Number of classes: {len(train_dataset.classes)}")
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(val_dataset):,}")
    print(f"Test samples: {len(test_dataset):,}")
    print(f"Total samples: {len(train_dataset) + len(val_dataset) + len(test_dataset):,}")
    
    return train_loader, val_loader, test_loader, train_dataset.classes

# =============================================================================
# MODEL CREATION
# =============================================================================
def create_model(num_classes):
    """
    Create ResNet50 model with transfer learning.
    
    Architecture:
    - Base: ResNet50 pretrained on ImageNet (IMAGENET1K_V2 weights)
    - Transfer Learning: Full fine-tuning (all layers trainable)
    - Custom Classifier: Final FC layer replaced for 39 disease classes
    
    Args:
        num_classes: Number of output classes (39 for plant diseases)
    
    Returns:
        model: ResNet50 model with custom classifier
    """
    print("\n" + "="*70)
    print("CREATING MODEL: ResNet50 with Transfer Learning")
    print("="*70)
    
    # Load pretrained ResNet50
    model = models.resnet50(weights='IMAGENET1K_V2')  # Pretrained on ImageNet
    
    # Replace the final fully connected layer for our 39 classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    # Note: All layers are trainable (full fine-tuning)
    # To freeze early layers, uncomment the following:
    # for param in list(model.parameters())[:-10]:
    #     param.requires_grad = False
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model: ResNet50")
    print(f"Pretrained weights: ImageNet (IMAGENET1K_V2)")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Output classes: {num_classes}")
    
    return model

# =============================================================================
# TRAINING FUNCTION
# =============================================================================
def train_model(model, train_loader, val_loader, num_epochs):
    """
    Train the model with validation monitoring.
    
    Training Details:
    - Loss Function: CrossEntropyLoss
    - Optimizer: Adam (learning rate: 0.001)
    - Learning Rate Scheduler: StepLR (reduces LR by 0.1x every 7 epochs)
    - Early Stopping: Saves best model based on validation accuracy
    
    Returns:
        train_losses: List of training losses per epoch
        train_accuracies: List of training accuracies per epoch
        val_losses: List of validation losses per epoch
        val_accuracies: List of validation accuracies per epoch
        best_val_acc: Best validation accuracy achieved
    """
    print("\n" + "="*70)
    print("TRAINING MODEL")
    print("="*70)
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Optimizer: Adam")
    print(f"Scheduler: StepLR (step_size=7, gamma=0.1)")
    print("="*70 + "\n")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # ===== TRAINING PHASE =====
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # ===== VALIDATION PHASE =====
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct_val / total_val
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch summary
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'  Learning Rate: {current_lr:.6f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f'  ✓ Model saved! (Best Val Acc: {best_val_acc:.2f}%)')
        
        print('-' * 70)
    
    print(f"\n✓ Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    
    return train_losses, train_accuracies, val_losses, val_accuracies, best_val_acc

# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================
def evaluate_model(model, data_loader, class_names, dataset_name='Validation'):
    """
    Evaluate model and calculate comprehensive metrics.
    
    Metrics Calculated:
    - Accuracy
    - Precision, Recall, F1-score (macro and weighted averages)
    - Per-class precision, recall, F1-score
    - Confusion Matrix
    
    Returns:
        metrics_dict: Dictionary containing all metrics
    """
    print(f"\n" + "="*70)
    print(f"EVALUATING MODEL ON {dataset_name.upper()} SET")
    print("="*70)
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    # Macro averages (unweighted mean across classes)
    precision_macro = precision.mean()
    recall_macro = recall.mean()
    f1_macro = f1.mean()
    
    # Weighted averages (weighted by support)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    # Print results
    print(f'\n{dataset_name} Set Results:')
    print(f'  Accuracy: {accuracy * 100:.2f}%')
    print(f'\n  Macro Averages (unweighted mean across classes):')
    print(f'    Precision: {precision_macro:.4f}')
    print(f'    Recall: {recall_macro:.4f}')
    print(f'    F1-Score: {f1_macro:.4f}')
    print(f'\n  Weighted Averages (weighted by support):')
    print(f'    Precision: {precision_weighted:.4f}')
    print(f'    Recall: {recall_weighted:.4f}')
    print(f'    F1-Score: {f1_weighted:.4f}')
    
    # Classification report
    print(f'\n  Detailed Classification Report:')
    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Create metrics dictionary
    metrics_dict = {
        'dataset': dataset_name,
        'accuracy': float(accuracy),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'precision_weighted': float(precision_weighted),
        'recall_weighted': float(recall_weighted),
        'f1_weighted': float(f1_weighted),
        'num_samples': len(all_labels),
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }
    
    # Per-class metrics
    per_class_metrics = []
    for i, class_name in enumerate(class_names):
        per_class_metrics.append({
            'class': class_name,
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i])
        })
    metrics_dict['per_class_metrics'] = per_class_metrics
    
    return metrics_dict, cm, class_names

def save_metrics(metrics_dict, filename):
    """Save metrics to JSON and text files."""
    # Save as JSON (structured format)
    json_path = RESULTS_DIR / f"{filename}.json"
    with open(json_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"\n✓ Metrics saved to: {json_path}")
    
    # Save as text (human-readable)
    txt_path = RESULTS_DIR / f"{filename}.txt"
    with open(txt_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write(f"{metrics_dict['dataset']} Set Evaluation Metrics\n")
        f.write("="*70 + "\n\n")
        f.write(f"Accuracy: {metrics_dict['accuracy'] * 100:.2f}%\n\n")
        f.write("Macro Averages:\n")
        f.write(f"  Precision: {metrics_dict['precision_macro']:.4f}\n")
        f.write(f"  Recall: {metrics_dict['recall_macro']:.4f}\n")
        f.write(f"  F1-Score: {metrics_dict['f1_macro']:.4f}\n\n")
        f.write("Weighted Averages:\n")
        f.write(f"  Precision: {metrics_dict['precision_weighted']:.4f}\n")
        f.write(f"  Recall: {metrics_dict['recall_weighted']:.4f}\n")
        f.write(f"  F1-Score: {metrics_dict['f1_weighted']:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(metrics_dict['classification_report'])
    print(f"✓ Text metrics saved to: {txt_path}")

def plot_training_history(train_losses, train_accs, val_losses, val_accs, save_path):
    """Plot and save training history (loss and accuracy curves)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Plot losses
    ax1.plot(epochs, train_losses, label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracies
    ax2.plot(epochs, train_accs, label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training history plot saved to: {save_path}")

def plot_confusion_matrix(cm, class_names, save_path, dataset_name='Validation'):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, square=True, linewidths=0.5)
    plt.title(f'Confusion Matrix - {dataset_name} Set', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to: {save_path}")

# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================
if __name__ == '__main__':
    print("\n" + "="*70)
    print("PLANT DISEASE CLASSIFICATION - MODEL TRAINING")
    print("Dataset: PlantVillage")
    print("Model: ResNet50 with Transfer Learning")
    print("Classes: 39 plant disease classes")
    print("="*70)
    
    # Step 1: Prepare dataset (create splits if needed)
    prepare_dataset_if_needed()
    
    # Step 2: Load data
    train_loader, val_loader, test_loader, class_names = load_data()
    
    # Step 3: Create model
    model = create_model(len(class_names))
    model = model.to(device)
    
    # Step 4: Train model
    train_losses, train_accs, val_losses, val_accs, best_val_acc = train_model(
        model, train_loader, val_loader, NUM_EPOCHS
    )
    
    # Step 5: Plot training history
    print("\n" + "="*70)
    print("GENERATING TRAINING PLOTS")
    print("="*70)
    training_plot_path = RESULTS_DIR / 'training_history.png'
    plot_training_history(train_losses, train_accs, val_losses, val_accs, training_plot_path)
    
    # Step 6: Load best model and evaluate on validation set
    print("\n" + "="*70)
    print("EVALUATING ON VALIDATION SET")
    print("="*70)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    val_metrics, val_cm, _ = evaluate_model(model, val_loader, class_names, 'Validation')
    save_metrics(val_metrics, 'validation_metrics')
    val_cm_path = RESULTS_DIR / 'confusion_matrix_validation.png'
    plot_confusion_matrix(val_cm, class_names, val_cm_path, 'Validation')
    
    # Step 7: Evaluate on test set (FINAL EVALUATION)
    print("\n" + "="*70)
    print("FINAL EVALUATION ON TEST SET")
    print("="*70)
    test_metrics, test_cm, _ = evaluate_model(model, test_loader, class_names, 'Test')
    save_metrics(test_metrics, 'test_metrics')
    test_cm_path = RESULTS_DIR / 'confusion_matrix_test.png'
    plot_confusion_matrix(test_cm, class_names, test_cm_path, 'Test')
    
    # Step 8: Save summary
    summary = {
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model': 'ResNet50',
        'pretrained_weights': 'ImageNet (IMAGENET1K_V2)',
        'num_classes': len(class_names),
        'hyperparameters': {
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'num_epochs': NUM_EPOCHS,
            'optimizer': 'Adam',
            'scheduler': 'StepLR (step_size=7, gamma=0.1)'
        },
        'dataset': {
            'name': 'PlantVillage',
            'train_samples': len(train_loader.dataset),
            'val_samples': len(val_loader.dataset),
            'test_samples': len(test_loader.dataset)
        },
        'best_validation_accuracy': float(best_val_acc),
        'validation_metrics': {
            'accuracy': val_metrics['accuracy'],
            'f1_macro': val_metrics['f1_macro'],
            'f1_weighted': val_metrics['f1_weighted']
        },
        'test_metrics': {
            'accuracy': test_metrics['accuracy'],
            'f1_macro': test_metrics['f1_macro'],
            'f1_weighted': test_metrics['f1_weighted']
        },
        'model_path': str(MODEL_SAVE_PATH),
        'results_directory': str(RESULTS_DIR)
    }
    
    summary_path = RESULTS_DIR / 'training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Training summary saved to: {summary_path}")
    
    # Final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nModel saved to: {MODEL_SAVE_PATH}")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"\nFinal Test Set Performance:")
    print(f"  Accuracy: {test_metrics['accuracy'] * 100:.2f}%")
    print(f"  F1-Score (Macro): {test_metrics['f1_macro']:.4f}")
    print(f"  F1-Score (Weighted): {test_metrics['f1_weighted']:.4f}")
    print("\n" + "="*70 + "\n")
