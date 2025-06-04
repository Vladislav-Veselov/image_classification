import pytest
import torch
from src.data import get_loaders

def test_data_loader_shapes():
    """Test that data loaders return correct shapes and types."""
    train_loader, val_loader, test_loader, classes = get_loaders(batch_size=32, val_ratio=0.1)
    
    # Test batch shapes
    train_batch, train_labels = next(iter(train_loader))
    val_batch, val_labels = next(iter(val_loader))
    test_batch, test_labels = next(iter(test_loader))
    
    assert train_batch.shape == (32, 3, 32, 32)  # batch_size, channels, height, width
    assert val_batch.shape == (32, 3, 32, 32)
    assert test_batch.shape == (32, 3, 32, 32)
    
    # Test label shapes and types
    assert train_labels.shape == (32,)
    assert val_labels.shape == (32,)
    assert test_labels.shape == (32,)
    assert train_labels.dtype == torch.long
    assert val_labels.dtype == torch.long
    assert test_labels.dtype == torch.long

def test_data_loader_ranges():
    """Test that data is properly normalized."""
    train_loader, val_loader, test_loader, _ = get_loaders(batch_size=32)
    
    for loader in [train_loader, val_loader, test_loader]:
        batch, _ = next(iter(loader))
        # Check if data is roughly normalized (mean close to 0, std close to 1)
        mean = batch.mean()
        std = batch.std()
        assert abs(mean) < 0.5  # Should be close to 0
        assert 0.5 < std < 2.0  # Should be roughly around 1

def test_train_val_split():
    """Test that train and validation sets are properly split."""
    train_loader, val_loader, _, _ = get_loaders(batch_size=32, val_ratio=0.1)
    
    # Check dataset sizes
    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    n_total = n_train + n_val
    
    assert n_total == 50000  # CIFAR10 training set size
    assert abs(n_val / n_total - 0.1) < 0.001  # val_ratio should be approximately 0.1

def test_augmentation():
    """Test that training data is augmented but validation/test are not."""
    train_loader, val_loader, test_loader, _ = get_loaders(batch_size=32)
    
    # Get same image twice from training set (should be different due to augmentation)
    train_img1, _ = train_loader.dataset[0]
    train_img2, _ = train_loader.dataset[0]
    assert not torch.allclose(train_img1, train_img2)
    
    # Get same image twice from validation set (should be identical)
    val_img1, _ = val_loader.dataset[0]
    val_img2, _ = val_loader.dataset[0]
    assert torch.allclose(val_img1, val_img2)
    
    # Get same image twice from test set (should be identical)
    test_img1, _ = test_loader.dataset[0]
    test_img2, _ = test_loader.dataset[0]
    assert torch.allclose(test_img1, test_img2)

def test_deterministic_split():
    """Test that the train/val split is deterministic with same seed."""
    # Get two sets of loaders with same seed
    loaders1 = get_loaders(batch_size=32, seed=42)
    loaders2 = get_loaders(batch_size=32, seed=42)
    
    # Compare train and val indices
    for (_, val1, _, _), (_, val2, _, _) in zip([loaders1], [loaders2]):
        assert len(val1.dataset) == len(val2.dataset)
        
        # Compare validation samples (should be identical since no augmentation)
        for i in range(5):
            img1, label1 = val1.dataset[i]
            img2, label2 = val2.dataset[i]
            assert torch.allclose(img1, img2)
            assert label1 == label2

def test_classes():
    """Test that class names are correct."""
    _, _, _, classes = get_loaders()
    expected_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
    assert classes == expected_classes
