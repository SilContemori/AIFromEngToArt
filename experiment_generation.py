"""Compare three local Tiny-SD settings without changing either model."""

import json
from time import perf_counter

import config
from generator import build_generation_prompt, generate_image, load_generator


def save_comparison_figure(records, output_path):
    """Save equally sized panels with images, settings and measured metrics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    figure, axes = plt.subplots(1, len(records), figsize=(15, 6), squeeze=False)
    for axis, record in zip(axes[0], records):
        with Image.open(record["generated_image_path"]) as image:
            axis.imshow(image)
        axis.set_title(
            f'{record["configuration"]}: {record["width"]} × {record["height"]}, '
            f'{record["steps"]} steps\n{record["generation_seconds"]:.2f} s | '
            f'{record["generated_predicted_style"]}\n'
            f'Impressionism score: {record["style_consistency_score"]:.2%}'
        )
        axis.axis("off")
    figure.suptitle(config.EXPERIMENT_SUBJECT + "\nTarget: " + config.EXPERIMENT_STYLE)
    figure.tight_layout(rect=(0, 0, 1, 0.86))
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main():
    """Load once, generate A/B/C with shared seed/guidance, classify, and report."""
    from classifier import load_classifier, predict_style

    config.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    generator = load_generator()
    classifier = load_classifier()
    load_seconds = perf_counter() - started
    records = []
    prompt = build_generation_prompt(config.EXPERIMENT_SUBJECT, config.EXPERIMENT_STYLE)
    report_path = config.EXPERIMENTS_DIR / "comparison.json"
    for label, width, height, steps in config.EXPERIMENT_CONFIGURATIONS:
        path = config.EXPERIMENTS_DIR / f"{label}-{width}x{height}-{steps}steps.png"
        started = perf_counter()
        generate_image(
            config.EXPERIMENT_SUBJECT, config.EXPERIMENT_STYLE, generator,
            prompt=prompt, output_path=path, width=width, height=height, steps=steps,
            seed=config.GENERATOR_SEED, guidance_scale=config.GENERATOR_GUIDANCE_SCALE,
        )
        generation_seconds = perf_counter() - started
        style, probabilities = predict_style(path, classifier)
        records.append({
            "configuration": label, "width": width, "height": height, "steps": steps,
            "seed": config.GENERATOR_SEED, "guidance_scale": config.GENERATOR_GUIDANCE_SCALE,
            "generation_seconds": generation_seconds, "generated_image_path": str(path),
            "generated_predicted_style": style, "generated_probabilities": probabilities,
            "style_consistency_score": float(probabilities[config.EXPERIMENT_STYLE]),
            "style_match": style == config.EXPERIMENT_STYLE,
        })
        # Persist each completed result in case a later generation is interrupted.
        report = {
            "model": config.GENERATOR_MODEL_ID, "device": config.GENERATOR_DEVICE,
            "subject": config.EXPERIMENT_SUBJECT, "target_style": config.EXPERIMENT_STYLE,
            "prompt": prompt, "model_load_seconds": load_seconds,
            "timing": "Generation and PNG saving; excludes model loading and classification. No warm-up; A runs first.",
            "results": records,
        }
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(records[-1]), flush=True)
    save_comparison_figure(records, config.EXPERIMENTS_DIR / "comparison.png")


if __name__ == "__main__":
    main()
