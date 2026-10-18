"""Connect local classification, style-conditioned generation and evaluation."""

import csv
import json
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import config
from generator import build_generation_prompt, generate_image, load_generator


def compute_style_consistency(target_style, generated_probabilities):
    """Return the generated image's softmax score for the original style."""
    return float(generated_probabilities[target_style])


@lru_cache(maxsize=1)
def get_classifier():
    """Load the existing classifier once per process for repeated UI requests."""
    from classifier import load_classifier
    return load_classifier(config.CHECKPOINT_PATH)


def run_styleloop(input_image_path, subject, **generation_settings):
    """Run both models, save PNG/JSON artifacts, and return measured results."""
    started = perf_counter()
    # Reject invalid inputs before loading either model.
    build_generation_prompt(subject, next(iter(config.STYLE_FOLDERS)))
    input_path = Path(input_image_path).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {input_path}")
    from PIL import Image
    with Image.open(input_path) as image:
        image.verify()

    from classifier import predict_style
    classifier = get_classifier()
    original_style, original_probabilities = predict_style(input_path, classifier)
    prompt = build_generation_prompt(subject, original_style)
    settings = {
        "width": config.GENERATOR_WIDTH, "height": config.GENERATOR_HEIGHT,
        "steps": config.GENERATOR_STEPS, "guidance_scale": config.GENERATOR_GUIDANCE_SCALE,
        "seed": config.GENERATOR_SEED,
    }
    if generation_settings.keys() - settings.keys():
        raise ValueError("Unknown generation setting")
    settings.update({key: value for key, value in generation_settings.items() if value is not None})
    run_id = uuid4().hex
    generated_path = config.GENERATED_DIR / f"styleloop-{run_id}.png"
    generate_image(
        subject.strip(), original_style, load_generator(),
        prompt=prompt, output_path=generated_path, **settings,
    )
    generated_style, generated_probabilities = predict_style(generated_path, classifier)
    result = {
        "input_image_path": str(input_path),
        "original_predicted_style": original_style,
        "original_probabilities": original_probabilities,
        "subject": subject.strip(),
        "generated_prompt": prompt,
        "generation_settings": settings,
        "generated_image_path": str(generated_path),
        "generated_predicted_style": generated_style,
        "generated_probabilities": generated_probabilities,
        "style_match": original_style == generated_style,
        "style_consistency_score": compute_style_consistency(
            original_style, generated_probabilities
        ),
        "execution_time_seconds": perf_counter() - started,
    }
    config.STYLELOOP_DIR.mkdir(parents=True, exist_ok=True)
    report_path = config.STYLELOOP_DIR / f"styleloop-{run_id}.json"
    report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved report: {report_path}")
    return result


def select_test_image():
    """Select the first test record and verify its prepared pixel hash."""
    from dataset import image_digest
    with (config.PREPARED_DATA_DIR / "index.csv").open(newline="", encoding="utf-8") as file:
        for record in csv.DictReader(file):
            if record["split"] == "test":
                path = (config.RAW_DATA_DIR / record["path"]).resolve()
                if not path.is_relative_to(config.RAW_DATA_DIR.resolve()):
                    raise ValueError("Test image path is outside the raw dataset")
                if image_digest(path) != record["sha256"]:
                    raise ValueError(f"Test image differs from prepared dataset: {path}")
                return path
    raise ValueError("Prepared index contains no test images")


def main():
    """Run the demo on a verified member of the existing test split."""
    result = run_styleloop(select_test_image(), config.STYLELOOP_DEMO_SUBJECT)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
