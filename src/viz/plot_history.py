import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

def plot_training_history(history, save_dir: Path):
    """Plot training history and save figures.
    
    Args:
        history: Dictionary containing training metrics
        save_dir: Directory to save the plots
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    fig.suptitle('Training History', fontsize=16)
    
    ax1.plot(history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(history['val_loss'], label='Validation Loss', marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.grid(True)
    ax1.legend()
    
    ax2.plot(history['train_acc'], label='Train Accuracy', marker='o')
    ax2.plot(history['val_acc'], label='Validation Accuracy', marker='o')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy Curves')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    with open(save_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=4)

def update_history(history, epoch, train_loss, train_acc, val_loss, val_acc):
    """Update the history dictionary with new metrics.
    
    Args:
        history: Dictionary containing training metrics
        epoch: Current epoch number
        train_loss: Training loss for current epoch
        train_acc: Training accuracy for current epoch
        val_loss: Validation loss for current epoch
        val_acc: Validation accuracy for current epoch
    
    Returns:
        Updated history dictionary
    """
    if history is None:
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    history['train_loss'].append(float(train_loss))
    history['train_acc'].append(float(train_acc))
    history['val_loss'].append(float(val_loss))
    history['val_acc'].append(float(val_acc))
    
    return history
