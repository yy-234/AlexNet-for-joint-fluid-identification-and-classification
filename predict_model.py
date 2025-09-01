import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib

# Set matplotlib to use English
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

# Define output directory
OUTPUT_DIR = '/guanjieye/model_final_10M_post10/'

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define class names for 6 classifications (English)
class_names = ['Class1', 'Class2', 'Class3', 'Class4', 'Class5', 'Class6']

class PredictionDataset(Dataset):
    """Prediction dataset"""
    def __init__(self, label_file, transform=None):
        """
        Args:
            label_file: Label file path containing image paths and labels
            transform: Image transformation
        """
        self.transform = transform
        self.data = []
        
        # Read label file
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    parts = line.split('\t')
                    if len(parts) == 2:
                        image_path = parts[0]
                        label = int(parts[1])
                        self.data.append({
                            'image_path': image_path,
                            'label': label
                        })
        
        print(f"Loaded {len(self.data)} samples from label file")
        
        # Show class distribution
        label_counts = {}
        for item in self.data:
            label = item['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print("Data class distribution:")
        for label in sorted(label_counts.keys()):
            print(f"  Class {label}: {label_counts[label]} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item['image_path']
        label = item['label']
        
        # Load image
        try:
            if not os.path.exists(image_path):
                print(f"Image file does not exist: {image_path}")
                # Return a black image as placeholder
                image = Image.new('RGB', (80, 80), (0, 0, 0))
            else:
                image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Cannot load image {image_path}: {e}")
            # Return a black image as placeholder
            image = Image.new('RGB', (80, 80), (0, 0, 0))
        
        # Apply transformation
        if self.transform:
            image = self.transform(image)
        
        return image, label, image_path

def create_model():
    model = models.alexnet(pretrained=True)
    model.classifier[1] = nn.Linear(9216, 512)  # First fully connected layer
    model.classifier[4] = nn.Linear(512, 512)   # Second fully connected layer
    model.classifier[6] = nn.Linear(512, 6)     # Last fully connected layer
    
    return model

def get_transforms():
    """Get image transformations"""
    return transforms.Compose([
        transforms.Resize(100),  # First resize to 100*100
        transforms.CenterCrop(80),  # Center crop to 80*80
        transforms.ToTensor(),  # Convert to tensor, automatically scale 0-255 to 0-1
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Scale 0-1 to (-1,1)
    ])

def get_preferred_device():
    """Get preferred GPU device"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"Detected {gpu_count} GPUs")
        
        # Prefer GPU choose your gpu list
        preferred_gpus = [, ,]
        for gpu_id in preferred_gpus:
            if gpu_id < gpu_count:
                device = torch.device(f"cuda:{gpu_id}")
                print(f"Selected GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
                return device
        
        # If preferred GPUs are not available, use first available GPU
        device = torch.device("cuda:0")
        print(f"Preferred GPUs not available, using GPU 0: {torch.cuda.get_device_name(0)}")
        return device
    else:
        device = torch.device("cpu")
        print("No GPU detected, using CPU")
        return device

def predict_images(model, data_loader, device):
    """Predict images"""
    model.eval()
    predictions = []
    image_paths = []
    probabilities = []
    true_labels = []
    
    with torch.no_grad():
        for images, labels, paths in data_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            predictions.extend(predicted.cpu().numpy())
            image_paths.extend(paths)
            probabilities.extend(probs.cpu().numpy())
            true_labels.extend(labels.numpy())
    
    return predictions, image_paths, probabilities, true_labels

def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics"""
    # Ensure labels start from 0 and have correct range
    y_true = np.array(y_true) - 1 if np.min(y_true) > 0 else np.array(y_true)
    y_pred = np.array(y_pred) - 1 if np.min(y_pred) > 0 else np.array(y_pred)
    
    # Ensure labels are in 0-5 range
    y_true = np.clip(y_true, 0, 5)
    y_pred = np.clip(y_pred, 0, 5)
    
    print(f"Processed label range - True: {np.min(y_true)}-{np.max(y_true)}, Predicted: {np.min(y_pred)}-{np.max(y_pred)}")
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(6))
    
    # Calculate metrics for each class
    class_metrics = {}
    for i in range(len(class_names)):
        # Binary confusion matrix (current class vs other classes)
        tp = cm[i, i]  # True positive
        fp = np.sum(cm[:, i]) - tp  # False positive
        fn = np.sum(cm[i, :]) - tp  # False negative
        tn = np.sum(cm) - tp - fp - fn  # True negative
        
        # Calculate metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity (recall)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        class_metrics[class_names[i]] = {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1_score': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        }
    
    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'confusion_matrix': cm,
        'class_metrics': class_metrics
    }

def main():
    """Main function"""
    # Set paths
    model_path = '/final_model.pth'
    label_file = 'label.txt'
    prediction_output = '/prediction.txt'
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Error: Model file not found {model_path}")
        return
    
    if not os.path.exists(label_file):
        print(f"Error: Label file not found {label_file}")
        return
    
    # Get device
    device = get_preferred_device()
    
    # Load trained model directly
    print("Loading trained model...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model and load weights
    model = create_model()
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"Model loaded successfully (Epoch {checkpoint['epoch']+1}, Validation accuracy: {checkpoint['val_accuracy']:.2f}%)")
    
    # Create dataset and data loader
    transform = get_transforms()
    dataset = PredictionDataset(label_file, transform)
    data_loader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    # Perform prediction
    print("Starting prediction...")
    predictions, image_paths, probabilities, true_labels_data = predict_images(model, data_loader, device)
    
    # Convert prediction results to 1-6 labels
    predictions_1_6 = [pred + 1 for pred in predictions]
    true_labels_1_6 = true_labels_data  # True labels are already 1-6, no need to +1
    
    # Add debug information
    print(f"Predicted label range: {min(predictions_1_6)} - {max(predictions_1_6)}")
    print(f"True label range: {min(true_labels_1_6)} - {max(true_labels_1_6)}")
    print(f"Predicted label distribution: {dict(zip(*np.unique(predictions_1_6, return_counts=True)))}")
    print(f"True label distribution: {dict(zip(*np.unique(true_labels_1_6, return_counts=True)))}")
    
    # Save prediction results to file
    with open(prediction_output, 'w', encoding='utf-8') as f:
        f.write("Image Path\tTrue Label\tPredicted Label\tTrue Class\tPredicted Class\n")
        for i, (path, true_label, pred) in enumerate(zip(image_paths, true_labels_1_6, predictions_1_6)):
            # Add boundary check
            true_class = class_names[true_label-1] if 1 <= true_label <= 6 else f"Class{true_label}"
            pred_class = class_names[pred-1] if 1 <= pred <= 6 else f"Class{pred}"
            f.write(f"{path}\t{true_label}\t{pred}\t{true_class}\t{pred_class}\n")
    
    print(f"Prediction results saved to: {prediction_output}")
    
    # Show prediction statistics
    pred_counts = {}
    for pred in predictions_1_6:
        pred_counts[pred] = pred_counts.get(pred, 0) + 1
    
    print(f"\nPrediction result statistics:")
    for label in sorted(pred_counts.keys()):
        print(f"  Class {label} ({class_names[label-1]}): {pred_counts[label]} samples")
    
    # Calculate evaluation metrics
    print(f"\n=== Calculating Evaluation Metrics ===")
    
    # Calculate metrics
    metrics = calculate_metrics(true_labels_1_6, predictions_1_6)
    
    print(f"\n=== Evaluation Results ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro Average Precision: {metrics['precision_macro']:.4f}")
    print(f"Macro Average Recall: {metrics['recall_macro']:.4f}")
    print(f"Macro Average F1 Score: {metrics['f1_macro']:.4f}")
    
    print(f"\nDetailed metrics for each class:")
    for class_name in class_names:
        class_metric = metrics['class_metrics'][class_name]
        print(f"{class_name}:")
        print(f"  Sensitivity (Recall): {class_metric['sensitivity']:.4f}")
        print(f"  Specificity: {class_metric['specificity']:.4f}")
        print(f"  Precision: {class_metric['precision']:.4f}")
        print(f"  F1 Score: {class_metric['f1_score']:.4f}")
    
    # Show confusion matrix
    print(f"\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    for i in range(6):
        row_str = " ".join([f"{cm[i,j]:3d}" for j in range(6)])
        print(f"Class{i}: {row_str}")
    
    # Save evaluation results
    eval_output = os.path.join(OUTPUT_DIR, 'evaluation_results.txt')
    with open(eval_output, 'w', encoding='utf-8') as f:
        f.write("label5m Data Prediction Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Number of evaluation samples: {len(true_labels_1_6)}\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Macro Average Precision: {metrics['precision_macro']:.4f}\n")
        f.write(f"Macro Average Recall: {metrics['recall_macro']:.4f}\n")
        f.write(f"Macro Average F1 Score: {metrics['f1_macro']:.4f}\n\n")
        
        f.write("Detailed metrics for each class:\n")
        for class_name in class_names:
            class_metric = metrics['class_metrics'][class_name]
            f.write(f"\n{class_name}:\n")
            f.write(f"  Sensitivity (Recall): {class_metric['sensitivity']:.4f}\n")
            f.write(f"  Specificity: {class_metric['specificity']:.4f}\n")
            f.write(f"  Precision: {class_metric['precision']:.4f}\n")
            f.write(f"  F1 Score: {class_metric['f1_score']:.4f}\n")
            f.write(f"  True Positive (TP): {class_metric['tp']}\n")
            f.write(f"  False Positive (FP): {class_metric['fp']}\n")
            f.write(f"  False Negative (FN): {class_metric['fn']}\n")
            f.write(f"  True Negative (TN): {class_metric['tn']}\n")
        
        f.write(f"\nConfusion Matrix:\n")
        for i in range(6):
            row_str = " ".join([f"{cm[i,j]:3d}" for j in range(6)])
            f.write(f"Class{i}: {row_str}\n")
    
    print(f"Evaluation results saved to: {eval_output}")
    
    # Save metrics to Excel
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Macro Average Precision', 'Macro Average Recall', 'Macro Average F1 Score'],
        'Value': [metrics['accuracy'], metrics['precision_macro'], 
               metrics['recall_macro'], metrics['f1_macro']]
    })
    metrics_excel = os.path.join(OUTPUT_DIR, 'evaluation_metrics.xlsx')
    metrics_df.to_excel(metrics_excel, index=False)
    print(f"Evaluation metrics saved to: {metrics_excel}")
    
    # Save prediction probabilities
    prob_output = os.path.join(OUTPUT_DIR, 'prediction_probabilities.xlsx')
    prob_df = pd.DataFrame(probabilities, columns=class_names)
    prob_df['image_path'] = image_paths
    prob_df['true_label'] = true_labels_1_6
    prob_df['predicted_label'] = predictions_1_6
    prob_df['true_class'] = [class_names[label-1] if 1 <= label <= 6 else f"Class{label}" for label in true_labels_1_6]
    prob_df['predicted_class'] = [class_names[pred-1] if 1 <= pred <= 6 else f"Class{pred}" for pred in predictions_1_6]
    prob_df.to_excel(prob_output, index=False)
    print(f"Prediction probabilities saved to: {prob_output}")
    
    print(f"\n=== Prediction Complete ===")
    print("Generated files:")
    print(f"- {prediction_output}: Prediction results")
    print(f"- {prob_output}: Prediction probabilities")
    print(f"- {eval_output}: Evaluation report")
    print(f"- {metrics_excel}: Evaluation metrics")

if __name__ == "__main__":
    main() 
