"""Small, local style-preservation experiment using the existing models."""

import csv
import json
from collections import Counter
from pathlib import Path
from time import perf_counter

import config
from generator import build_generation_prompt, generate_image, load_generator
from pipeline import compute_style_consistency


def select_inputs(records, predict, digest):
    """Select up to three correct test inputs per class in sorted path order."""
    selected = {style: [] for style in config.STYLE_FOLDERS}
    for record in sorted(records, key=lambda row: row["path"]):
        style = record["style"]
        if record["split"] != "test" or len(selected[style]) >= len(config.PHASE7_SUBJECTS):
            continue
        path = (config.RAW_DATA_DIR / record["path"]).resolve()
        if not path.is_relative_to(config.RAW_DATA_DIR.resolve()):
            raise ValueError("Test input is outside the raw dataset")
        if digest(path) != record["sha256"]:
            raise ValueError(f"Test input differs from prepared data: {path}")
        predicted, probabilities = predict(path)
        if predicted == style:
            selected[style].append({
                "input_image_path": str(path), "input_true_style": style,
                "input_predicted_style": predicted, "input_probabilities": probabilities,
            })
    return selected


def summarize(results):
    """Compute counted aggregates, retaining empty classes as null metrics."""
    def metrics(rows):
        count = len(rows)
        return {
            "sample_count": count,
            "style_match_rate": sum(row["style_match"] for row in rows) / count if count else None,
            "average_style_consistency_score": sum(row["style_consistency_score"] for row in rows) / count if count else None,
            "average_generation_seconds": sum(row["generation_seconds"] for row in rows) / count if count else None,
        }
    styles = list(config.STYLE_FOLDERS)
    return {
        "overall": metrics(results),
        "per_style": {style: metrics([row for row in results if row["input_predicted_style"] == style]) for style in styles},
        "predicted_style_distribution": {
            "sample_count": len(results),
            "counts": {style: Counter(row["generated_predicted_style"] for row in results)[style] for style in styles},
        },
        "confusion_matrix": {
            "labels": styles, "sample_count": len(results),
            "rows_target_columns_predicted": [[sum(row["input_predicted_style"] == target and row["generated_predicted_style"] == predicted for row in results) for predicted in styles] for target in styles],
        },
    }


def export_results(results, report, directory):
    """Persist full records and a flat CSV with all five output probabilities."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    fields = ["input_image_path", "input_true_style", "input_predicted_style", "subject", "generated_prompt", "generated_image_path", "generated_predicted_style", "style_match", "style_consistency_score", "generation_seconds"]
    probability_fields = [f"probability_{style}" for style in config.STYLE_FOLDERS]
    with (directory / "results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields + probability_fields)
        writer.writeheader()
        for row in results:
            writer.writerow({**{field: row[field] for field in fields}, **{f"probability_{style}": row["generated_probabilities"][style] for style in config.STYLE_FOLDERS}})


def save_figures(results, summary, directory):
    """Draw a counted confusion matrix, match-rate bars and grouped contact sheet."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    styles = list(config.STYLE_FOLDERS)
    matrix = summary["confusion_matrix"]["rows_target_columns_predicted"]
    figure, axis = plt.subplots(figsize=(9, 8))
    axis.imshow(matrix, cmap="Blues", vmin=0)
    axis.set(xticks=range(5), yticks=range(5), xticklabels=styles, yticklabels=styles,
             xlabel="Generated predicted style", ylabel="Input predicted style (target)")
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right")
    for i in range(5):
        for j in range(5):
            axis.text(j, i, str(matrix[i][j]), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(directory / "confusion_matrix.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    rates = [summary["per_style"][style]["style_match_rate"] for style in styles]
    bars = axis.bar(styles, [rate if rate is not None else 0 for rate in rates])
    for bar, style, rate in zip(bars, styles, rates):
        count = summary["per_style"][style]["sample_count"]
        axis.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                  f'{rate:.0%} (n={count})' if rate is not None else "N/A (n=0)", ha="center")
    axis.set(ylim=(0, 1.15), ylabel="Style match rate")
    plt.setp(axis.get_xticklabels(), rotation=20, ha="right")
    figure.tight_layout()
    figure.savefig(directory / "style_match_rates.png", dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(5, 3, figsize=(13, 21))
    for i, style in enumerate(styles):
        rows = [row for row in results if row["input_predicted_style"] == style]
        for j, axis in enumerate(axes[i]):
            axis.axis("off")
            if j >= len(rows):
                axis.set_title(f"{style}\nNo available correct input")
                continue
            row = rows[j]
            with Image.open(row["generated_image_path"]) as image:
                axis.imshow(image)
            axis.set_title(f'{style} — subject {j+1}\nPredicted: {row["generated_predicted_style"]}\nScore: {row["style_consistency_score"]:.1%} | match: {row["style_match"]}', fontsize=10)
    figure.suptitle("Phase 7: garden / city / boat (columns); classifier scores are uncalibrated")
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(directory / "contact_sheet.png", dpi=120)
    plt.close(figure)


def main(resume=False):
    """Select verified inputs, run at most 15 generations once, and export evidence."""
    from classifier import load_classifier, predict_style
    from dataset import image_digest

    directory = config.PHASE7_DIR
    directory.mkdir(parents=True, exist_ok=True)
    # Prevent accidental overwrites or extra generations; reproduction uses a fresh folder.
    if any(directory.iterdir()) and not resume:
        raise FileExistsError(f"Experiment directory must be empty: {directory}")
    classifier = load_classifier(device="cpu")
    with (config.PREPARED_DATA_DIR / "index.csv").open(newline="", encoding="utf-8") as file:
        selected = select_inputs(list(csv.DictReader(file)), lambda path: predict_style(path, classifier), image_digest)
    settings = {"width": config.GENERATOR_WIDTH, "height": config.GENERATOR_HEIGHT,
                "steps": config.GENERATOR_STEPS, "guidance_scale": config.GENERATOR_GUIDANCE_SCALE,
                "seed": config.GENERATOR_SEED}
    report = {
        "status": "running", "model": config.GENERATOR_MODEL_ID, "classifier_device": "cpu",
        "generation_device": config.GENERATOR_DEVICE, "generation_settings": settings,
        "subjects": config.PHASE7_SUBJECTS, "selected_inputs": selected,
        "selection": "First three correct, hash-verified test inputs per class, sorted by relative path; paired with subjects in order. Fewer correct inputs means fewer generations.",
        "timing": "Generation and PNG saving only; excludes model loading and both classifications. No warm-up.",
        "limitations": [
            "Classifier test accuracy is 62%; predictions are imperfect.",
            "Consistency scores are uncalibrated softmax scores, not probabilities of artistic correctness.",
            "Using the same classifier for selection and evaluation can introduce bias; correctly classified inputs are a selected subset.",
            "Classifier evaluation does not replace human artistic judgment; a match need not be visually convincing.",
            "Small exploratory experiment, not a statistically conclusive benchmark; subjects are repeated within each style.",
            "Fixed seed supports reproduction on the same software and hardware, not guaranteed cross-platform pixel identity.",
        ],
        "shortfalls": {style: 3-len(rows) for style, rows in selected.items() if len(rows) < 3},
    }
    results = []
    if resume:
        previous = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        for key in ("generation_settings", "selected_inputs", "subjects", "model"):
            if previous[key] != json.loads(json.dumps(report[key])):
                raise ValueError(f"Cannot resume: changed {key}")
        results = previous["results"]
        expected_pairs = [(row["input_image_path"], subject)
                          for inputs in selected.values()
                          for row, subject in zip(inputs, config.PHASE7_SUBJECTS)]
        if [(row["input_image_path"], row["subject"]) for row in results] != expected_pairs[:len(results)]:
            raise ValueError("Saved results do not match the experiment plan")
        if any(not Path(row["generated_image_path"]).is_file() for row in results):
            raise ValueError("A completed image is missing")
    export_results(results, {**report, **summarize(results), "results": results}, directory)
    generator = load_generator()
    for style_index, inputs in enumerate(selected.values()):
        for subject_index, (original, subject) in enumerate(zip(inputs, config.PHASE7_SUBJECTS)):
            if any(row["input_image_path"] == original["input_image_path"] and row["subject"] == subject for row in results):
                continue
            target = original["input_predicted_style"]
            prompt = build_generation_prompt(subject, target)
            path = directory / f"style-{style_index+1}-subject-{subject_index+1}.png"
            if path.exists():
                raise FileExistsError(f"Unrecorded image exists; inspect manually: {path}")
            started = perf_counter()
            generate_image(subject, target, generator, prompt=prompt, output_path=path, **settings)
            seconds = perf_counter() - started
            predicted, probabilities = predict_style(path, classifier)
            results.append({**original, "subject": subject, "generated_prompt": prompt,
                            "generated_image_path": str(path), "generated_predicted_style": predicted,
                            "generated_probabilities": probabilities, "style_match": target == predicted,
                            "style_consistency_score": compute_style_consistency(target, probabilities),
                            "generation_seconds": seconds})
            export_results(results, {**report, **summarize(results), "results": results}, directory)
            print(f"Completed {len(results)}: {target} -> {predicted} ({seconds:.2f}s)", flush=True)
    report.update(summarize(results))
    report.update(status="complete", results=results)
    export_results(results, report, directory)
    save_figures(results, report, directory)
    print(json.dumps(report["overall"], indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true", help="Explicitly continue unstarted pairs without regenerating saved images")
    main(resume=parser.parse_args().resume)
