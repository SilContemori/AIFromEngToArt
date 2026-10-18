"""Prepare manual review materials from saved Phase 7 images; no model calls."""

import csv
import json
from pathlib import Path
import textwrap

import config


def load_results():
    """Read the completed experiment and resolve its images inside this checkout."""
    report = json.loads((config.PHASE7_DIR / "summary.json").read_text())
    results = report["results"]
    expected = {(style, subject) for style in config.STYLE_FOLDERS for subject in config.PHASE7_SUBJECTS}
    pairs = [(row["input_predicted_style"], row["subject"]) for row in results]
    if report["status"] != "complete" or len(results) != 15 or set(pairs) != expected:
        raise ValueError("Expected a completed experiment with 15 unique style/subject pairs")
    for row in results:
        # Historical reports contain runtime absolute paths; images live beside the report.
        path = config.PHASE7_DIR / Path(row["generated_image_path"]).name
        if not path.is_file():
            raise FileNotFoundError(path)
        row["generated_image_path"] = path.relative_to(config.PROJECT_ROOT).as_posix()
    return results


def save_template(results):
    """Write measured fields and blank subjective fields; preserve existing reviews."""
    path = config.PHASE8_DIR / "visual_evaluation.csv"
    fields = ["Target artistic style", "Generation subject", "Generated image path",
              "Classifier predicted style", "Classifier consistency score",
              "Subject recognizable: Yes / Partially / No",
              "Visual style coherence: High / Medium / Low", "Notes"]
    if path.exists():
        print(f"Preserved existing review: {path}")
        return
    with path.open("x", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(fields)
        for row in results:
            writer.writerow([row["input_predicted_style"], row["subject"], row["generated_image_path"],
                             row["generated_predicted_style"], row["style_consistency_score"], "", "", ""])


def save_contact_sheet(results):
    """Label all existing images with subjects, filenames and measured predictions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    figure, axes = plt.subplots(5, 3, figsize=(18, 29))
    for i, style in enumerate(config.STYLE_FOLDERS):
        for j, subject in enumerate(config.PHASE7_SUBJECTS):
            row = next(row for row in results if (row["input_predicted_style"], row["subject"]) == (style, subject))
            path = config.PROJECT_ROOT / row["generated_image_path"]
            with Image.open(path) as image:
                image.load()
                if image.size != (512, 512):
                    raise ValueError(f"Unexpected image dimensions: {path}")
                axes[i, j].imshow(image)
            axes[i, j].axis("off")
            axes[i, j].set_title(
                f'Target: {style}\n' + textwrap.fill(subject, 40) +
                f'\nPredicted: {row["generated_predicted_style"]}\n'
                f'Target score: {row["style_consistency_score"]:.2%} | Match: {"Yes" if row["style_match"] else "No"}\n{path.name}',
                fontsize=12, pad=10)
    figure.suptitle("StyleLoop — 15 saved Phase 7 images\nClassifier scores are uncalibrated; human evaluation is pending", fontsize=18)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(config.PHASE8_DIR / "contact_sheet.png", dpi=140)
    plt.close(figure)


def main():
    """Create the review folder, CSV template and labeled contact sheet."""
    results = load_results()
    config.PHASE8_DIR.mkdir(parents=True, exist_ok=True)
    save_template(results)
    save_contact_sheet(results)
    print(f"Prepared {len(results)} images for manual review in {config.PHASE8_DIR}")


if __name__ == "__main__":
    main()
