import torch
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import json
from tqdm import tqdm
import sys
from pathlib import Path

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data import get_loaders
from src.train import load_model, evaluate

def load_best_model(model_name: str, device: str = "cuda"):
    """Load the best model checkpoint for a given model type."""
    exp_dir = Path("experiments") / f"{model_name}_baseline"
    checkpoint = torch.load(exp_dir / "best_model.pt", map_location=device, weights_only=True)
    model = load_model(model_name).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint['val_acc']

def evaluate_model(model, test_loader, device: str = "cuda"):
    """Evaluate model and return predictions, labels, and metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

def plot_confusion_matrix(y_true, y_pred, classes, model_name, save_dir: Path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_dir / f'confusion_matrix_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_model_comparison(metrics_df, save_dir: Path):
    """Create comparison plots for different models."""
    plt.figure(figsize=(10, 6))
    sns.barplot(data=metrics_df, x='Model', y='Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.ylim(0, 100)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    out_dir = Path("experiments/_model_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    _, _, test_loader, classes = get_loaders(batch_size=128)
    
    metrics = []
    all_results = {}
    
    model_types = ["lenet", "custom_cnn", "resnet", "resnet_adapted"]
    
    for model_name in model_types:
        print(f"\nEvaluating {model_name}...")
        try:
            model, best_val_acc = load_best_model(model_name, device)
            
            predictions, labels, probabilities = evaluate_model(model, test_loader, device)
            test_acc, _ = evaluate(model, test_loader, device)  # Only get accuracy
            
            metrics.append({
                'Model': model_name,
                'Accuracy': test_acc,
                'Best Val Accuracy': best_val_acc
            })
            
            all_results[model_name] = {
                'test_accuracy': float(test_acc),
                'best_val_accuracy': float(best_val_acc),
                'classification_report': classification_report(
                    labels, predictions, 
                    target_names=classes, 
                    output_dict=True
                )
            }
            
            plot_confusion_matrix(labels, predictions, classes, model_name, out_dir)
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {str(e)}")
            continue
    
    if metrics:
        metrics_df = pd.DataFrame(metrics)
        plot_model_comparison(metrics_df, out_dir)
        
        metrics_df.to_csv(out_dir / 'model_metrics.csv', index=False)
        with open(out_dir / 'detailed_results.json', 'w') as f:
            json.dump(all_results, f, indent=4)
        
        print("\nModel Comparison Summary:")
        print(metrics_df.to_string(index=False))
    else:
        print("No models were successfully evaluated.")

if __name__ == "__main__":
    main()
