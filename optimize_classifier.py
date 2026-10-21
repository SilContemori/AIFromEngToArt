"""Compare frozen ResNet18 heads on validation, then test the selected candidate."""

import csv
import hashlib
import json
import random
import shutil

import torch
from torch.utils.data import DataLoader

import config
from classifier import build_model, get_device, load_classifier
from dataset import PaintingDataset, verify_records
from train import calculate_metrics, run_epoch


def selection_key(accuracy, loss):
    """Prefer validation accuracy, using lower loss only to break ties."""
    return accuracy, -loss


def train_trial(records, device, learning_rate, directory, index_digest):
    """Train one reproducible head with early stopping and save its best weights."""
    random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)
    generator = torch.Generator().manual_seed(config.RANDOM_SEED)
    loaders = {
        split: DataLoader(PaintingDataset(records, split), batch_size=config.BATCH_SIZE,
                          shuffle=split == "train", num_workers=config.NUM_WORKERS,
                          generator=generator if split == "train" else None)
        for split in ("train", "validation")
    }
    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=learning_rate)
    best_key = (-1, float("-inf"))
    stale_epochs = 0
    history = []
    checkpoint_path = directory / f"lr-{learning_rate}.pt"
    for epoch in range(1, config.CLASSIFIER_MAX_EPOCHS + 1):
        train_loss, train_accuracy, _, _ = run_epoch(model, loaders["train"], device, optimizer)
        loss, accuracy, _, _ = run_epoch(model, loaders["validation"], device)
        row = dict(epoch=epoch, train_loss=train_loss, train_accuracy=train_accuracy,
                   validation_loss=loss, validation_accuracy=accuracy)
        history.append(row)
        print(json.dumps(dict(learning_rate=learning_rate, **row)), flush=True)
        key = selection_key(accuracy, loss)
        if key > best_key:
            best_key = key
            stale_epochs = 0
            best = dict(epoch=epoch, validation_accuracy=accuracy, validation_loss=loss,
                        learning_rate=learning_rate, seed=config.RANDOM_SEED,
                        index_sha256=index_digest, classes=list(config.STYLE_FOLDERS),
                        device=str(device), batch_size=config.BATCH_SIZE,
                        max_epochs=config.CLASSIFIER_MAX_EPOCHS,
                        patience=config.CLASSIFIER_PATIENCE,
                        preprocessing="resize_224_square_imagenet_normalization")
            torch.save(dict(best, model_state={name: value.detach().cpu().clone()
                                              for name, value in model.state_dict().items()}),
                       checkpoint_path)
        else:
            stale_epochs += 1
        (directory / f"lr-{learning_rate}-history.json").write_text(json.dumps(history, indent=2))
        if stale_epochs >= config.CLASSIFIER_PATIENCE:
            break
    return dict(best, checkpoint=checkpoint_path.name, epochs_run=len(history))


def main():
    """Preserve baseline, compare on validation and test the selected candidate once."""
    directory = config.CLASSIFIER_EXPERIMENT_DIR
    # Refuse to overwrite experiment evidence or a previous baseline backup.
    directory.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(config.CPU_THREADS)
    device = get_device()
    index_path = config.PREPARED_DATA_DIR / "index.csv"
    with index_path.open() as file:
        records = list(csv.DictReader(file))
    verify_records(records, config.RAW_DATA_DIR)
    index_digest = hashlib.sha256(index_path.read_bytes()).hexdigest()
    baseline_path = directory / "baseline.pt"
    shutil.copy2(config.CHECKPOINT_PATH, baseline_path)
    baseline_metadata = torch.load(baseline_path, map_location="cpu", weights_only=True)
    if baseline_metadata["index_sha256"] != index_digest:
        raise ValueError("Baseline and experiment must use the same dataset index")
    validation_loader = DataLoader(PaintingDataset(records, "validation"),
                                   batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)
    baseline = load_classifier(baseline_path, device)
    loss, accuracy, _, _ = run_epoch(baseline, validation_loader, device)
    baseline_result = dict(validation_accuracy=accuracy, validation_loss=loss,
                           checkpoint="baseline.pt")
    del baseline, baseline_metadata
    trials = []
    for learning_rate in config.CLASSIFIER_TRIAL_LEARNING_RATES:
        trials.append(train_trial(records, device, learning_rate, directory, index_digest))
        (directory / "trials.json").write_text(json.dumps(trials, indent=2))
    selected = max([baseline_result, *trials],
                   key=lambda row: selection_key(row["validation_accuracy"], row["validation_loss"]))
    # Selection is complete before accessing test images or test metrics.
    model = load_classifier(directory / selected["checkpoint"], device)
    test_loader = DataLoader(PaintingDataset(records, "test"), batch_size=config.BATCH_SIZE,
                             num_workers=config.NUM_WORKERS)
    test_loss, _, targets, predictions = run_epoch(model, test_loader, device)
    metrics = calculate_metrics(targets, predictions)
    metrics["test_loss"] = test_loss
    (directory / "selected_test_metrics.json").write_text(json.dumps(metrics, indent=2))
    report = dict(device=str(device), index_sha256=index_digest, baseline=baseline_result,
                  trials=trials, selected=selected, promoted=False, test_metrics=metrics,
                  limitations=["The existing test set was previously evaluated; this is not a new independent test.",
                               "No artist-separated split; no conclusions about diffusion quality."])
    (directory / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
