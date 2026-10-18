"""Train only the ResNet18 head; select on validation, then evaluate on test."""

import csv
import hashlib
import json
import random
import time

import torch
from torch.utils.data import DataLoader

import config
from classifier import build_model, get_device, load_classifier
from dataset import PaintingDataset, verify_records


def run_epoch(model, loader, device, optimizer=None):
    """Run training or evaluation, returning sample-weighted loss and accuracy."""
    # eval() keeps frozen BatchNorm running statistics unchanged during training.
    model.eval()
    total_loss = correct = count = 0
    targets, predictions = [], []
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if optimizer is not None:
                optimizer.zero_grad()
            logits = model(images)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            predicted = logits.argmax(dim=1)
            total_loss += loss.item() * len(labels)
            correct += (predicted == labels).sum().item()
            count += len(labels)
            targets.extend(labels.cpu().tolist())
            predictions.extend(predicted.cpu().tolist())
    return total_loss / count, correct / count, targets, predictions


def calculate_metrics(targets, predictions):
    """Calculate test accuracy, per-class and macro/weighted precision, recall, F1."""
    classes = list(config.STYLE_FOLDERS)
    matrix = [[0] * len(classes) for _ in classes]
    for actual, predicted in zip(targets, predictions):
        matrix[actual][predicted] += 1
    per_class = {}
    for index, style in enumerate(classes):
        true_positive = matrix[index][index]
        support = sum(matrix[index])
        predicted_count = sum(row[index] for row in matrix)
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[style] = dict(precision=precision, recall=recall, f1=f1, support=support)
    metrics = {"accuracy": sum(matrix[i][i] for i in range(len(classes))) / len(targets),
               "per_class": per_class, "confusion_matrix": matrix, "classes": classes}
    for average in ("macro", "weighted"):
        metrics[average] = {name: sum(row[name] * (row["support"] if average == "weighted" else 1)
                                    for row in per_class.values()) /
                           (len(targets) if average == "weighted" else len(classes))
                           for name in ("precision", "recall", "f1")}
    return metrics


def save_graphs(history, metrics):
    """Save training/validation curves and the final test confusion matrix."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name in ("train_loss", "validation_loss"):
        axes[0].plot(epochs, [row[name] for row in history], label=name)
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy loss")
    axes[0].legend()
    axes[1].plot(epochs, [row["validation_accuracy"] for row in history])
    axes[1].set(xlabel="Epoch", ylabel="Validation accuracy", ylim=(0, 1))
    figure.tight_layout()
    figure.savefig(config.OUTPUTS_DIR / "training_curves.png")
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(9, 8))
    axis.imshow(metrics["confusion_matrix"], cmap="Blues")
    axis.set(xticks=range(5), yticks=range(5), xticklabels=metrics["classes"],
             yticklabels=metrics["classes"], xlabel="Predicted", ylabel="Actual",
             title="Test confusion matrix")
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right")
    for row in range(5):
        for column in range(5):
            axis.text(column, row, metrics["confusion_matrix"][row][column], ha="center")
    figure.tight_layout()
    figure.savefig(config.OUTPUTS_DIR / "test_confusion_matrix.png")
    plt.close(figure)


def main():
    """Verify existing data, train and select the best head, then evaluate test once."""
    random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)
    torch.set_num_threads(config.CPU_THREADS)
    device = get_device()
    print(f"Device: {device}", flush=True)
    index_path = config.PREPARED_DATA_DIR / "index.csv"
    with index_path.open() as file:
        records = list(csv.DictReader(file))
    verify_records(records, config.RAW_DATA_DIR)
    generator = torch.Generator().manual_seed(config.RANDOM_SEED)
    loaders = {split: DataLoader(PaintingDataset(records, split), batch_size=config.BATCH_SIZE,
                                shuffle=split == "train", num_workers=config.NUM_WORKERS,
                                generator=generator if split == "train" else None)
               for split in ("train", "validation", "test")}
    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=config.LEARNING_RATE)
    config.MODELS_DIR.mkdir(exist_ok=True)
    config.OUTPUTS_DIR.mkdir(exist_ok=True)
    history, best_accuracy = [], -1
    for epoch in range(1, config.EPOCHS + 1):
        started = time.monotonic()
        train_loss, _, _, _ = run_epoch(model, loaders["train"], device, optimizer)
        validation_loss, accuracy, _, _ = run_epoch(model, loaders["validation"], device)
        row = dict(epoch=epoch, train_loss=train_loss, validation_loss=validation_loss,
                   validation_accuracy=accuracy, seconds=time.monotonic() - started)
        history.append(row)
        print(json.dumps(row), flush=True)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({"model_state": {key: value.cpu() for key, value in model.state_dict().items()},
                        "classes": list(config.STYLE_FOLDERS), "epoch": epoch,
                        "validation_accuracy": accuracy,
                        "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
                        "seed": config.RANDOM_SEED, "device": str(device),
                        "epochs": config.EPOCHS, "batch_size": config.BATCH_SIZE,
                        "learning_rate": config.LEARNING_RATE}, config.CHECKPOINT_PATH)
        (config.OUTPUTS_DIR / "training_history.json").write_text(json.dumps(history, indent=2))
    selected = load_classifier(device=device)
    test_loss, _, targets, predictions = run_epoch(selected, loaders["test"], device)
    metrics = calculate_metrics(targets, predictions)
    metrics["test_loss"] = test_loss
    metrics["selected_epoch"] = torch.load(config.CHECKPOINT_PATH, weights_only=True)["epoch"]
    (config.OUTPUTS_DIR / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    save_graphs(history, metrics)
    print("Final test metrics: " + json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
