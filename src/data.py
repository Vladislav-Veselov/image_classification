from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import torch
from typing import Tuple, List

def get_loaders(batch_size: int = 128,
                val_ratio: float = 0.1,
                seed: int = 42,
                num_workers: int = 4,
                pin_memory: bool = True
               ) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:

    g = torch.Generator().manual_seed(seed)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    # two dataset objects that point to the same 50 000 images
    train_aug_ds = CIFAR10("data", train=True,  download=True,  transform=transform_train)
    val_plain_ds = CIFAR10("data", train=True,  download=False, transform=transform_eval)

    test_ds      = CIFAR10("data", train=False, download=True,  transform=transform_eval)

    n_total = len(train_aug_ds)        
    n_val   = int(n_total * val_ratio)
    perm    = torch.randperm(n_total, generator=g)
    idx_val, idx_train = perm[:n_val], perm[n_val:]

    train_subset = Subset(train_aug_ds, idx_train)
    val_subset   = Subset(val_plain_ds, idx_val)

    train_loader = DataLoader(train_subset, batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    val_loader   = DataLoader(val_subset,   batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_ds,      batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)

    print("Dataset sizes:")
    print(f"  Training:   {len(train_subset)} images (augmented, shuffled)")
    print(f"  Validation: {len(val_subset)} images (plain, deterministic)")
    print(f"  Test:       {len(test_ds)} images  (plain, deterministic)")

    return train_loader, val_loader, test_loader, train_aug_ds.classes
