"""Frozen ResNet18 construction, checkpoint loading, and style inference."""

from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision.models import resnet18

import config
from dataset import get_transform


def get_device():
    """Select Apple MPS when available, otherwise CPU."""
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def build_model(pretrained=True):
    """Freeze ImageNet ResNet18 and replace its head with five trainable outputs."""
    # Always construct offline; load the original 1000-class weights before replacing the head.
    model = resnet18(weights=None)
    if pretrained:
        state = torch.load(config.PRETRAINED_WEIGHTS_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.fc = torch.nn.Linear(model.fc.in_features, len(config.STYLE_FOLDERS))
    return model


def load_classifier(checkpoint_path=config.CHECKPOINT_PATH, device=None):
    """Load saved weights without downloading ImageNet weights; validate label order."""
    device = get_device() if device is None else torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["classes"] != list(config.STYLE_FOLDERS):
        raise ValueError("Checkpoint class order differs from configured class mapping")
    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval()


def predict_style(image, model=None, checkpoint_path=config.CHECKPOINT_PATH):
    """Accept a PIL image or path and return the winning style and all probabilities."""
    if model is None:
        model = load_classifier(checkpoint_path)
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
    else:
        image = ImageOps.exif_transpose(image).convert("RGB")
    inputs = get_transform()(image).unsqueeze(0).to(next(model.parameters()).device)
    model.eval()
    with torch.inference_mode():
        probabilities = model(inputs).softmax(dim=1)[0].cpu().tolist()
    scores = dict(zip(config.STYLE_FOLDERS, probabilities))
    return max(scores, key=scores.get), scores
