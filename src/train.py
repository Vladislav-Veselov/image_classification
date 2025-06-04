import argparse, torch, yaml
from pathlib import Path
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from src.data import get_loaders
from importlib import import_module
from tqdm import tqdm
from src.viz.plot_history import plot_training_history, update_history

def load_model(name: str, num_classes: int = 10):
    module = import_module(f"src.models.{name}")
    model_map = {
        "lenet": "LeNet",
        "custom_cnn": "CustomCNN",
        "resnet": "ResNet18"
    }
    model_class = model_map.get(name)
    if model_class is None:
        raise ValueError(f"Unknown model: {name}. Available models: {list(model_map.keys())}")
    return getattr(module, model_class)(num_classes)

def evaluate(model, loader, device="cpu", criterion=None):
    model.eval()
    correct = total = 0
    total_loss = 0
    with torch.no_grad():
        pbar = tqdm(loader, desc="Evaluating", leave=False)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            if criterion is not None:
                total_loss += criterion(outputs, labels).item()
            preds = outputs.argmax(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            pbar.set_postfix({"acc": f"{100.0 * correct / total:.2f}%"})
    
    avg_loss = total_loss / len(loader) if criterion is not None else 0
    accuracy = 100.0 * correct / total
    return accuracy, avg_loss

def main(cfg_path: str, model_name: str = None):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_name = model_name or cfg["model"]
    
    val_ratio = cfg.get("val_ratio", 0.1)
    train_loader, val_loader, test_loader, classes = get_loaders(
        batch_size=cfg["batch_size"],
        val_ratio=val_ratio,
        seed=cfg.get("seed", 42)
    )
    
    model = load_model(model_name).to(device)
    optimizer = SGD(model.parameters(), lr=cfg["lr"], momentum=0.9)
    criterion = CrossEntropyLoss()

    best_acc = 0
    patience = 5
    patience_counter = 0
    out_dir = Path("experiments") / cfg["experiment_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    
    history = None

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['epochs']}", 
                   leave=True,
                   dynamic_ncols=True,
                   mininterval=0.1)
        total_loss = 0
        correct = total = 0
        batch_times = []
        
        for i, (imgs, labels) in enumerate(pbar):
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            end_time.record()
            
            preds = outputs.argmax(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            total_loss += loss.item()
            
            torch.cuda.synchronize()
            
            batch_time = start_time.elapsed_time(end_time) / 1000
            batch_times.append(batch_time)
            avg_loss = total_loss / (i + 1)
            avg_time = sum(batch_times[-100:]) / min(len(batch_times), 100)
            train_acc = 100.0 * correct / total
            
            pbar.set_postfix({
                "loss": f"{avg_loss:.3f}",
                "acc": f"{train_acc:.2f}%",
                "time/batch": f"{avg_time:.3f}s",
                "imgs/s": f"{cfg['batch_size']/avg_time:.1f}"
            })
        
        train_loss = total_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        
        val_acc, val_loss = evaluate(model, val_loader, device, criterion)
        
        history = update_history(history, epoch, train_loss, train_acc, val_loss, val_acc)
        
        print(f"Epoch {epoch}:")
        print(f"  Train - Loss: {train_loss:.3f}, Accuracy: {train_acc:.2f}%")
        print(f"  Val   - Loss: {val_loss:.3f}, Accuracy: {val_acc:.2f}%")
        print(f"  Time  - Avg batch: {sum(batch_times)/len(batch_times):.3f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, out_dir / "best_model.pt")
            print(f"New best model saved with accuracy: {val_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs. Best accuracy: {best_acc:.2f}%")
            
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch} epochs!")
            print(f"Best validation accuracy: {best_acc:.2f}%")
            break

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
        }, out_dir / f"epoch{epoch}.pt")
        
        plot_training_history(history, out_dir / "plots")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the config file")
    parser.add_argument("--model", help="Override the model specified in config (e.g., lenet, resnet)")
    args = parser.parse_args()
    main(args.config, args.model)

