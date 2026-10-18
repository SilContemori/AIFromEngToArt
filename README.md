# StyleLoop

University project for **AI From Engineering to Arts**. The application
classifies painting styles, generates a new image from a style and subject, and
classifies the result to evaluate style consistency.

## Current scope

Phases 1–6 implement local dataset preparation, a frozen-backbone ResNet18
classifier, local Tiny-SD generation, and the end-to-end StyleLoop pipeline.
Models are loaded from local files; datasets are never downloaded. Phase 6.5 adds
style descriptions, adjustable generation settings and a controlled Tiny-SD comparison.
Phase 7 completed 15 exploratory generations; Phase 8 prepares final review materials.

## Setup and execution

Use uv exclusively, with Python 3.12 selected by `.python-version`. Dependencies
are in `pyproject.toml`, with exact versions in `uv.lock`. Pillow validates images;
torchvision and PyTorch provide image transforms and tensors.

From the repository root:

```sh
uv sync --locked
uv run python dataset.py
uv run python -m unittest discover -s tests -v
```

If the dependencies are already cached, the entire phase can run offline:

```sh
uv sync --locked --offline
uv run --offline python dataset.py
uv run --offline python -m unittest discover -s tests -v
```

Preparation reads only local files. It overwrites `data/prepared/index.csv` and
`data/prepared/report.json` after successful validation. It does not change raw
images. Missing folders or fewer than 200 valid unique images in a class cause
an error rather than duplicate or substitute images. Audit counts are printed
before a shortage error; outputs from an earlier successful run are not removed.

## Local source and usage conditions

The user manually downloaded a Hugging Face dataset described as **WikiArt 21
Styles** and extracted it into `data/raw/`. The actual extraction has **27 style
folders**. The precise repository/version cannot be established from the folder
names alone; do not assume this is the same release as a similarly named mirror.
The [21-style mirror card](https://huggingface.co/datasets/michaelrodcs/wikiart-21-styles-optimized)
labels that mirror CC BY-NC 4.0 and describes educational/non-commercial use.
That statement does not authenticate this local extraction's provenance.

The [upstream WikiArt release](https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset)
restricts use to non-commercial research and requires compliance with
[WikiArt terms](https://www.wikiart.org/en/terms-of-use). Images remain subject to
original-author rights; WikiArt includes public-domain and copyrighted works.
Keep the source card, download URL/revision, and accompanying terms with your
project records and confirm the exact release before redistributing images.
This phase uses the user-provided local files and does not redistribute them.

Required folders:

```text
data/raw/
  Impressionism/
  Cubism/
  Expressionism/
  Abstract_Expressionism/
  Pop_Art/
```

Additional style folders are ignored. Downloaded data, generated indexes and
reports, models, and outputs are excluded from Git.

## Preparation and measured distribution

Sorted file paths plus seed 42 make selection and splitting reproducible for an
unchanged local dataset and locked environment. All supported files in the five
folders are fully decoded first. SHA-256 hashes of oriented RGB pixels and image
size identify exact visual duplicates even across filenames or lossless encodings.
Deduplication is global: the first occurrence in configured class/path order is
kept; later occurrences are excluded, including those in other classes. This does
not detect resized, cropped, or slightly modified copies. Splits are stratified
by style, not by artist; artist overlap can make later evaluation optimistic.

Measured from the local dataset:

| Style | Candidates | Corrupt | Duplicates excluded | Valid unique | Train | Validation | Test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Impressionism | 13060 | 0 | 7 | 13053 | 140 | 30 | 30 |
| Cubism | 2235 | 0 | 0 | 2235 | 140 | 30 | 30 |
| Expressionism | 6736 | 0 | 75 | 6661 | 140 | 30 | 30 |
| Abstract Expressionism | 2782 | 0 | 4 | 2778 | 140 | 30 | 30 |
| Pop Art | 1483 | 0 | 6 | 1477 | 140 | 30 | 30 |
| Selected total | | | | | 700 | 150 | 150 |

Exactly 1,000 images are indexed. Their original files occupy 14,677,346 bytes
(about 14.7 MB); no selected-image copies are created. The complete local raw
extraction occupies approximately 1.2 GB. No images were downloaded by the agent.
The CSV stores paths relative to `data/raw/`, style, pixel hash, and split.
The JSON stores the seed, counts, exclusions, distribution, and selected file bytes.

## Preprocessing

`load_image(path, training=False)` returns an RGB float tensor of shape
`(3, 224, 224)`, using torchvision resizing and ImageNet mean/std normalization
for future ResNet18 use. Resizing is performed on load, not on disk, and directly
resizes to a square for simplicity (aspect ratio may change).

Only `training=True` adds random horizontal flipping. Validation and test inputs
are deterministic. This phase does not save augmented images or train a model.
A future training loop must seed its own augmentation RNG; the preparation seed
currently controls subset selection and splits.

## Files and functions

- `config.py`: all directory paths, style names, target count, seed, split fractions,
  image size, normalization values, and supported image extensions.
- `dataset.py`:
  - `image_digest`: decode and hash oriented RGB image content.
  - `collect_images`: scan the five folders, reject corruption, and deduplicate.
  - `split_images`: sample each class and assign deterministic 70/15/15 splits.
  - `verify_records`: re-read selected images and reject changed content, duplicate
    hashes, unknown styles, or invalid split names.
  - `prepare_dataset`: run preparation, print counts, and write CSV/JSON outputs.
  - `get_transform`: build evaluation or training torchvision transforms.
  - `load_image`: read one local image and apply the selected transform.
- `tests/test_dataset.py`: temporary-image tests for corruption, duplicate content,
  shortages, missing folders, reproducibility, stratification, output files,
  tensor size, normalization, and training-only augmentation.
- `tests/test_smoke.py`: module import and configuration checks.
- `train.py`, `classifier.py`: Phase 3 training, evaluation and inference (below).
- `generator.py`, `pipeline.py`: local generation and the Phase 5 pipeline.
- `app.py`: Phase 6 local Gradio interface (see installation status below).
  The existing `main.py` is a PyCharm sample.
- `AGENTS.md`: development guidelines. `.gitignore`: local artifact exclusions.

To verify the real subset again without scanning all candidates:

```sh
uv run python - <<'PY'
import csv
import config
from dataset import verify_records, load_image
with (config.PREPARED_DATA_DIR / "index.csv").open() as file:
    records = list(csv.DictReader(file))
verify_records(records, config.RAW_DATA_DIR)
assert len(records) == 1000
print("Verified", len(records), "readable, unique images")
print("Example tensor:", tuple(load_image(config.RAW_DATA_DIR / records[0]["path"]).shape))
PY
```

## Phase 3: artistic style classifier

Phase 3 adds ImageNet-pretrained ResNet18 transfer learning. Only its final
512-to-5 linear layer is optimized with Adam and cross-entropy. The backbone
parameters and BatchNorm running statistics remain frozen. Existing Phase 2
resizing, normalization, training-only flips, CSV splits and class order are reused.
Matplotlib is added solely for saved graphs; metrics use simple confusion counts.
Generation and feedback were added in Phases 4–5; the interface source is in Phase 6.

```sh
uv sync --locked
uv run python -m unittest discover -s tests -v
uv run python train.py
```

Do not rerun dataset preparation to train. Training reads and verifies the existing
`data/prepared/index.csv`. Place the official pretrained ImageNet weights at
`models/pretrained/checkpoints/resnet18-f37072fd.pth` (approximately 45 MB).
`PRETRAINED_WEIGHTS_PATH` in `config.py` controls this local path. Model construction
always uses `weights=None`, then strictly loads this local state dictionary before
replacing the head. It never attempts a download; a missing or incompatible file
raises an error. Run `uv run --offline python train.py` with cached dependencies.
Each run overwrites the Phase 3 checkpoint and output files.

Settings are centralized in `config.py`: 10 epochs, batch size 16, learning rate
0.001, seed 42, zero DataLoader workers, four CPU threads. MPS is chosen when
available, otherwise CPU. No MPS fallback environment flags or experimental
operations are enabled. Python, PyTorch and the training shuffle are seeded;
exact numerical reproducibility across devices is not guaranteed.

Each epoch prints training loss, validation loss, validation accuracy and seconds.
The highest validation accuracy selects the checkpoint; ties keep the earlier
epoch. Only after all epochs is that selected checkpoint evaluated on test.
Test results never influence optimization or checkpoint selection. Macro metrics
average classes equally; weighted metrics weight by support. Undefined precision,
recall or F1 is recorded as zero. Existing artist overlap across splits remains
a limitation of this small style classification experiment.

Artifacts:

- `models/style_classifier.pt`: selected weights, class order, selected epoch,
  validation accuracy, index hash, device, seed and training settings.
- `outputs/training_history.json`: epoch training and validation results.
- `outputs/training_curves.png`: losses and validation accuracy.
- `outputs/test_metrics.json`: test loss, accuracy, per-class and macro/weighted
  precision, recall and F1, confusion counts, class order and selected epoch.
- `outputs/test_confusion_matrix.png`: actual rows and predicted columns.

Inference accepts a PIL image or image path:

```sh
uv run python - <<'PY'
from classifier import load_classifier, predict_style
model = load_classifier()
style, probabilities = predict_style("path/to/painting.jpg", model)
print(style)
print(probabilities)
PY
```

Softmax scores sum to one but are not calibrated confidence estimates.
Checkpoint loading does not download weights. It rejects mismatched class order.

New functions and dataset methods:

- `classifier.get_device`: choose MPS or CPU.
- `classifier.build_model`: load ResNet18, freeze its backbone, replace its head.
- `classifier.load_classifier`: restore a checkpoint, validate labels, set eval mode.
- `classifier.predict_style`: preprocess one image and return style and probabilities.
- `PaintingDataset.__init__`: select existing split records and label mapping.
- `PaintingDataset.__len__`: report split size.
- `PaintingDataset.__getitem__`: return transformed image and integer label.
  This small map-style class is needed by DataLoader for indexed image loading.
- `train.run_epoch`: run forward passes and optionally optimize the head; accumulate
  sample-weighted losses and accuracy plus labels/predictions.
- `train.calculate_metrics`: calculate confusion counts and classification metrics.
- `train.save_graphs`: write the requested curves and test confusion plot.
- `train.main`: seed RNGs, verify data, train, select, save and evaluate.

`tests/test_classifier.py` checks output shape, frozen parameters/statistics during
an optimizer step, checkpoint round-trip, label validation, PIL/path inference,
softmax probabilities and known metric counts. Tests use random weights offline;
they do not establish trained accuracy. Test methods describe their purpose in
docstrings. Actual training results are recorded separately in the output files.

### Local weights verification

The supplied weights loaded strictly into the original ResNet18. Their SHA-256 is
`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.
All 1,000 existing images passed content verification, with balanced 700/150/150
splits. Only `fc.weight` and `fc.bias` are trainable. The environment selects CPU
because MPS is unavailable. The local-loading test checks preserved backbone
weights, construction with `weights=None`, and failure on missing files.

### Completed training on 2026-10-17

`uv run --offline python train.py` completed all 10 epochs on CPU using only local
weights. The best validation checkpoint was epoch 10 (68.00% accuracy). It was
then evaluated once on the 150 test images: accuracy 62.00%, macro precision
63.32%, macro recall 62.00%, macro F1 61.99%, test loss 0.9236. Weighted averages
equal macro averages because each test class has 30 images. Full epoch results
and per-class metrics are in `outputs/training_history.json` and
`outputs/test_metrics.json`. The selected model and both graphs were saved at
the artifact paths above. All 11 automated tests passed with
`uv run --offline python -m unittest discover -s tests -v`.

## Phase 4B: local image generation

`generator.py` implements local text-to-image generation with the approved
[Segmind Tiny-SD model](https://huggingface.co/segmind/tiny-sd), licensed under
CreativeML OpenRAIL-M. Follow its use restrictions and redistribution conditions;
see the license linked from the official model card. Phase 5 connects classification and generation; Phase 6 adds the interface source. The classifier and its checkpoint are unchanged by this phase.

### Setup and download

Generation requires Diffusers (pipeline), Transformers (CLIP text encoder), and
Hugging Face Hub (snapshot download). Additional runtime dependencies are resolved
by uv. Accelerate is not needed: loading explicitly disables low-memory loading.
PyTorch 2.14.1 and torchvision 0.29.1 must remain installed.

The current `pyproject.toml` and `uv.lock` include the generation dependencies.
The local model snapshot is installed and the real Phase 5 CPU run succeeded.
With the existing environment and model files, run:

```sh
uv run --offline python generator.py
uv run --offline python pipeline.py
uv run --offline python -m unittest discover -s tests -v
```

On a fresh setup, use `uv sync --locked`, then `uv run python download_generator.py`
to obtain Tiny-SD before running offline. Preserve the existing classifier checkpoint.

The download script checks for at least 5 GiB free (59 GiB was observed here),
then uses `snapshot_download` to preserve repository folders inside
`models/pretrained/tiny-sd/`. Its allowlist selects the original `.bin` weights,
component configs, tokenizer, scheduler and available documentation, excluding
alternate weight formats and variants. Git already ignores `models/*` and
`outputs/*`. A failed download can be resumed by rerunning the same command.
Inference uses only the local directory and cannot download missing components.

### Functions and execution

- `download_generator.main()` checks free space and downloads the selected files.
- `generator.load_generator()` loads the local pipeline once per process via
  `lru_cache`, using CPU float32 and `local_files_only=True`.
- `generator.generate_image(subject, style, pipeline, *, prompt=None, output_path=None)` validates text, constructs
  a prompt from both values, generates one image, saves a uniquely named PNG in
  `outputs/generated/`, prints its absolute path, and returns the PIL image.

All settings live in `config.py`: the initial command uses “a small house near a
lake”, “Impressionism”, 256 × 256 pixels, 10 steps, guidance 7.5 and seed 42.
The command prints loading time, generation-plus-save time, image dimensions and
process peak RSS, and saves those measurements to `outputs/generated/last_run.json`.
Peak RSS includes loading and is not a measurement of total unified system memory.
A fixed seed improves repeatability but does not guarantee identical pixels across
library versions and platforms. PNG filenames remain unique across repeated runs.

### Verification and limitations

Mock-based generation tests verify configuration, saved images, and prompt handling
without loading diffusion weights. Phase 5 additionally ran the real local models,
as recorded below. Tiny-SD was trained at 512 pixels; this 256-pixel, 10-step CPU
configuration is a functionality test. Style adherence is not guaranteed. The model
snapshot supplies no safety checker. No advanced optimizations were introduced.

## Phase 5: end-to-end StyleLoop

Run from the repository root:

```sh
uv run --offline python pipeline.py
uv run --offline python -m unittest discover -s tests -v
```

The demo selects the first `test` row in the existing prepared index and verifies
its pixel SHA-256 against the preparation record. It does not rebuild or change the
splits. The checkpoint is loaded once and reused for both classifications, with
its existing five-class order. The original prediction directly determines the
prompt sent to the existing Tiny-SD generator. CPU, 256 × 256, 10 steps, seed 42
and guidance 7.5 remain configured in `config.py`.

For another image and subject:

```python
from pipeline import run_styleloop
result = run_styleloop("path/to/painting.jpg", "a small house near a lake")
```

Functions in `pipeline.py`:

- `build_generation_prompt(subject, style)`: validate text and combine the subject
  with a configured style in an English prompt.
- `compute_style_consistency(target_style, generated_probabilities)`: return the
  generated image's softmax score for the input image's predicted style.
- `run_styleloop(input_image_path, subject)`: validate the image, classify it,
  generate and save a PNG, classify that PNG, compute match/score, save a JSON
  report, and return the result dictionary.
- `select_test_image()`: choose and verify a genuine member of the prepared test split.
- `main()`: execute the configured demo and print the result dictionary.

`generator.generate_image` retains its original behavior by default. Optional
`prompt` and `output_path` arguments let StyleLoop pass and record the exact prompt
and associate the generated PNG with its report without duplicating generation logic.
No new dependencies or classifier changes were needed.

Each run saves uniquely named artifacts to `outputs/generated/styleloop-<id>.png`
and `outputs/styleloop/styleloop-<id>.json`. JSON includes input/output paths,
both predicted labels and all five probabilities, subject, prompt, style match,
style consistency score, and execution time in seconds. Timing includes validation,
model loading, both classifications, generation and PNG saving; it excludes JSON
serialization/writing and terminal printing. Existing reports are preserved.

Style match compares the two predicted labels. The consistency score is a
**classifier-based proxy for stylistic consistency**, not an objective measurement
of artistic quality or a calibrated probability of correctness. The classifier's
62% test accuracy and the small generation configuration limit interpretation.

### Actual run on 2026-10-17

- Input: `data/raw/Impressionism/mary-cassatt_mother-and-child-2.jpg`, verified
  against its test-split preparation record.
- Original prediction: Impressionism (softmax score 0.96672553).
- Subject: `a small house near a lake`.
- Prompt: `A painting of a small house near a lake, in the style of Impressionism`.
- Generated PNG: `outputs/generated/styleloop-dcaf9002008c40f7ae5a19066cc63c63.png`.
- Generated prediction: Pop Art (softmax score 0.70672041).
- Style match: **False**.
- Style consistency score: **0.19267024**.
- Execution time: **10.809233 seconds**.
- Report: `outputs/styleloop/styleloop-dcaf9002008c40f7ae5a19066cc63c63.json`.

This run establishes that the pipeline functions, but did not achieve a matching
style prediction. Diffusers emitted a non-fatal `torch_dtype` deprecation warning;
PyTorch also reported that multiprocessing redirects are unsupported on macOS.
The first sandboxed uv command could not access its cache; the authorized rerun
succeeded offline. No models were downloaded or retrained.

`tests/test_pipeline.py` covers prompt construction, consistency scores, matching
and nonmatching predictions, result structure and saved JSON, shared classifier
reuse, invalid paths/corrupt images, empty subjects, test-split selection, and changed
image rejection. Model calls are mocked. Gradio belongs to Phase 6 below.

Final verification: all **21 tests passed** with
`uv run --offline python -m unittest discover -s tests -v` (0.883 seconds).
An initial test compared a macOS temporary-directory alias with its resolved path;
the assertion now resolves both paths. No production change was required.

## Phase 6: Gradio interface

The interface reuses `pipeline.run_styleloop` for classification, generation,
reclassification and saved artifacts. Its two columns contain an artwork upload
and preview, subject, **Generate Artwork** button, original style and exact prompt
on the left; generated artwork, generated style, consistency percentage, style
match and execution time on the right. The score limitation is displayed below
the results. Only standard [Gradio components](https://gradio.app/docs/gradio/blocks)
are used, without custom JavaScript or CSS.

### Install and launch

Gradio is installed and recorded in `pyproject.toml` and `uv.lock`.
Use the locked environment from the repository root:

```sh
uv sync --locked
uv run --offline python app.py
```

The local
URL is configured as **http://127.0.0.1:7860**. It will be available after a successful
launch. No public sharing link or external model service is used. Stop the server
with Ctrl+C. Host and port live in `config.py`.

Upload a painting, enter a subject such as `a small house near a lake`, and click
**Generate Artwork**. The first generation loads the existing models; subsequent
requests reuse them. The classifier is cached by `pipeline.get_classifier`, and
Tiny-SD retains its existing generator cache. Restart the process after replacing
model files. Requests are queued one at a time to share the models safely.
Execution time is the pipeline's measured time, excluding queue wait and UI
rendering. Outputs still save to the Phase 5 PNG/JSON directories.

New functions:

- `app.interface_error(message)`: create a Gradio error for a user-facing message;
  Gradio is imported only when needed.
- `app.generate_artwork(input_image_path, subject)`: check required fields, call
  the existing pipeline, handle errors, and format its eight displayed outputs.
- `app.build_interface()`: construct the two-column layout and wire the button to
  the callback with single-request concurrency.
- `app.main()`: launch the local server on the configured host and port.
- `pipeline.get_classifier()`: load and cache the existing checkpoint once per process.

No classifier architecture, checkpoint, diffusion configuration or training was
changed. No experimental evaluation was introduced.

### Validation status

```sh
uv run --offline python -m unittest discover -s tests -v
```

Callback tests use mocked pipeline results and a lightweight mock of Gradio's
error object. They verify output mapping, percentage/time formatting, both match
outcomes, missing inputs, invalid images and model errors. Pipeline tests also
verify that repeated requests share one classifier load.

Historical Phase 6 startup failed before Gradio was installed. Phase 6.5
subsequently verified interface construction. See Phase 8 below for the current
startup verification and environment limitation.

All **25 automated tests passed** after the final changes. A real direct call to
`app.generate_artwork` also completed using the local models (no model mocks):
the verified test image `mary-cassatt_mother-and-child-2.jpg` was classified as
Impressionism; the generated image was classified as Pop Art. The callback
display values were **19.27%**, **No**, and **9.03 seconds**. This exercised the
callback and pipeline without starting Gradio or testing browser rendering.
Artifacts: `outputs/generated/styleloop-0e37ea0c98cd4ebaba657054347b7701.png` and
`outputs/styleloop/styleloop-0e37ea0c98cd4ebaba657054347b7701.json`.

## Phase 6.5: generation quality comparison

All five styles now have concise visual descriptions in `config.STYLE_DESCRIPTIONS`.
The input classifier's prediction still selects the description automatically.
`generator.build_generation_prompt(subject, style)` validates the subject/style and
constructs the shared prompt; `pipeline.build_generation_prompt` re-exports it for
existing callers. `generate_image` accepts optional `width`, `height`, `steps`,
`guidance_scale`, and `seed` keyword overrides, validates them before generation,
and otherwise uses `config.py`. `run_styleloop` forwards these settings and records
their effective values in its JSON report. Existing calls with two arguments work.

No dependencies, dataset changes, retraining, or model replacements are required.
To reproduce the experiment or run the interface:

```sh
uv run --offline python experiment_generation.py
uv run --offline python -m unittest discover -s tests -v
uv run --offline python app.py
```

`experiment_generation.main()` loads both existing local models once, generates
A/B/C sequentially with the configured fixed subject, Impressionism, seed 42 and
guidance 7.5, classifies each output, and writes `outputs/experiments/comparison.json`.
`save_comparison_figure(records, output_path)` saves a labeled side-by-side figure
using the existing Matplotlib dependency. The experiment overwrites its own named
PNGs, JSON and figure; ordinary pipeline runs retain unique filenames.

Generation times include denoising, decoding and PNG saving, and exclude model
loading and classification. Model loading is reported separately. There is no
warm-up; A runs first. This is one sample per setting, with no statistical timing
estimate. A fixed seed at different resolutions does not imply identical starting
noise or composition. Resolution and steps change together between A and B, so
their individual effects cannot be separated by this experiment. The prior user
test used a different prompt and is not a controlled prompt-only baseline.

Inspect the full-resolution PNGs as well as `outputs/experiments/comparison.png`.
Classifier softmax scores are uncalibrated proxies, and the existing classifier
has 62% test accuracy. Scores do not measure subject fidelity, anatomy or artistic
quality. No automatic setting selection or regeneration loop is implemented.

### Measured local results (2026-10-18)

| Configuration | Resolution | Steps | Generation + save | Predicted style | Impressionism score | Style match |
| --- | --- | ---: | ---: | --- | ---: | --- |
| A | 256 × 256 | 10 | 7.7942 s | Pop Art | 0.2061% | No |
| B | 512 × 512 | 20 | 47.4782 s | Pop Art | 44.4150% | No |
| C | 512 × 512 | 30 | 64.0024 s | Pop Art | 44.1116% | No |

Images: `outputs/experiments/A-256x256-10steps.png`,
`outputs/experiments/B-512x512-20steps.png`, and
`outputs/experiments/C-512x512-30steps.png`. The JSON retains full-precision times,
all five probabilities, the exact prompt and manual visual assessment.

Visual inspection: A shows the woman, bench and flowers, but has a soft,
photographic appearance. B has clearer painterly brushwork, natural light and a
recognizable seated woman in a colorful garden. C looks very similar to B;
facial and hand details remain simplified in both. More steps did not improve
the classifier score or produce an obvious visual benefit in this sample.

Recommended defaults, now in `config.py`: **512 × 512, 20 steps, guidance 7.5,
seed 42**. This recommendation combines visual inspection and latency, rather
than selecting a score winner automatically. It does not establish reliable
style matching: all three classifier labels still disagree with the target.

After the experiment, `app.build_interface()` gained a collapsed **Generation
settings** panel for square resolution (256/512), steps, guidance and seed.
`app.generate_artwork()` validates required inputs, forwards the optional settings
to the existing pipeline and formats the same eight outputs. Changing controls
does not change the input-predicted style used to construct the prompt.

Verification: all **28 automated tests passed** (0.832 s), including override
validation, shared experiment controls, report contents, pipeline settings
recording and UI forwarding. Gradio construction succeeded with 28 components
and six callback inputs. Browser upload/click interaction was not exercised in
this phase. Three real local Tiny-SD generations and classifier predictions were
executed separately from mocked tests; no classifier retraining occurred.

## Phase 7 — exploratory style evaluation

Run locally with the existing prepared test split and saved models:

```bash
uv run --offline python -m unittest discover -s tests -v
uv run --offline python experiment_evaluation.py
```

The output folder is `outputs/experiments/phase7/`. The runner requires an empty
folder to prevent accidental overwrites or additional generations. To reproduce,
archive the existing folder first (for example, move it to `phase7-previous`).
No models are trained, modified or downloaded.

Inputs are taken only from `data/prepared/index.csv` test records, sorted by
relative path, checked against their prepared pixel hashes, then classified with
the existing checkpoint on CPU. For each of the five styles, the first three
correctly classified inputs are paired in order with these subjects:

1. A woman sitting in a flower garden
2. A street in a European city at sunset
3. A small sailing boat on the sea

The **actual input prediction** supplies the generation style and prompt. Each
style/subject is generated once, at 512 × 512, 20 steps, guidance 7.5, seed 42.
Fewer than three correct inputs for a class means fewer paired subjects; shortfalls
are reported. This is text-conditioned generation, not image-to-image generation:
original image content does not condition Tiny-SD beyond the predicted label.
There are no retries, warm-up generations or discarded unsuccessful matches.

`summary.json` records selected paths, true and predicted input labels, input
probabilities, exact settings, full individual results, shortfalls and counted
aggregates. `results.csv` records each generation and all five output probabilities.
The confusion matrix has target styles as rows and generated predictions as
columns. `confusion_matrix.png`, `style_match_rates.png` and `contact_sheet.png`
show all results, grouped in configured style order; contact-sheet columns are
garden, city and boat. Generation timings include diffusion, decoding and PNG
saving, excluding model loading, input selection and reclassification.

New functions in `experiment_evaluation.py`:

- `select_inputs`: verify test images and select up to three correct inputs per class.
- `summarize`: calculate match rates, mean target scores, mean generation times,
  output-label counts and confusion counts, with sample sizes and null empty metrics.
- `export_results`: save flat CSV probabilities and detailed JSON after every result.
- `save_figures`: save the confusion matrix, match-rate bars and grouped contact sheet.
- `main`: load existing models once, run the unique pairs and write final artifacts.
  An interrupted run retains completed records; it is not automatically resumed.

The classifier has **62% test accuracy**. Style consistency is its uncalibrated
softmax score for the input-predicted style, not a calibrated probability of
artistic correctness. Using the same classifier to select inputs and evaluate
outputs can introduce bias, and correctly classified inputs form a selected
subset. Classifier evaluation does not replace human artistic judgment; a style
match does not necessarily mean an image is visually convincing. Fifteen images
with repeated subjects and one seed are a small exploratory experiment, not a
statistically conclusive benchmark. Fixed seeds support reproducibility on the
same environment; exact pixels are not guaranteed across hardware/library versions.

If a process is interrupted, `uv run --offline python experiment_evaluation.py --resume`
explicitly continues only pairs absent from the saved results, after checking the
settings, subjects and selected inputs. Completed images must still exist. An
unrecorded PNG blocks continuation for manual inspection; it is never overwritten.
This is explicit continuation, not an automatic retry or regeneration loop.

### Phase 7 measured results (2026-10-18)

All five classes supplied three correctly classified test inputs. All 15 unique
style/subject pairs completed with the requested defaults; no generated result
was discarded. Overall matches: **7/15 (46.67%)**. Mean generation and PNG-save
time: **32.06 seconds (n=15)**. Overall mean target score: **0.4274 (n=15)**.

| Target style | Samples | Matches | Match rate | Mean consistency score | Mean generation time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Impressionism | 3 | 1 | 33.33% | 0.4343 | 33.46 s |
| Cubism | 3 | 2 | 66.67% | 0.4915 | 28.44 s |
| Expressionism | 3 | 0 | 0.00% | 0.0883 | 30.07 s |
| Abstract Expressionism | 3 | 1 | 33.33% | 0.3982 | 33.13 s |
| Pop Art | 3 | 3 | 100.00% | 0.7249 | 35.19 s |

Output predictions (n=15): Impressionism 1, Cubism 2, Expressionism 0,
Abstract Expressionism 4, Pop Art 8. Pop Art dominated the classifier's output;
these data cannot separate generator behavior from classifier bias. Impressionism
matched for the garden subject, so the earlier mismatch with a different prompt
was not universal. This comparison does not establish a controlled improvement.

Visual inspection of the contact sheet: subjects remain recognizable, Cubism
outputs use geometric forms, and Pop Art outputs use flatter colors and outlines.
The Abstract Expressionism outputs remain representational despite the prompt's
nonrepresentational description. Expressionism and Abstract Expressionism share
similar compositions and brushwork. These observations support reviewing the
images alongside the classifier measurements, without changing any scores.

Execution note: the initial native process exited with status 139 after saving
two results and before producing the third PNG. One explicit `--resume` invocation
completed the remaining 13 pairs without regenerating completed outputs. Timing
measurements span two processes; model loading is excluded. There is no automatic
retry loop. Final verification passed: 15 unique pairs, 15 valid 512 × 512 PNGs,
15 CSV rows with all five probabilities, JSON aggregates and all three figures.
All **31 automated tests passed** (1.355 s). Test output is retained in
`outputs/experiments/phase7/tests.log`. Phase 7 stops here; no academic paper,
model change, retraining or interface change is included.

## Phase 8 — final presentation preparation

Read `PROJECT_SUMMARY.md` for the concise project description and
`PHASE8_REVIEW.md` for measured analysis, representative examples and repository
verification. No models or core pipeline behavior were changed.

```sh
uv sync --locked --offline
uv run --offline python prepare_visual_evaluation.py
uv run --offline python -m unittest discover -s tests -v
uv run --offline python app.py
```

The preparation command reads only the existing Phase 7 report and PNGs. It does
not load models or generate images. Outputs are
`outputs/experiments/phase8/visual_evaluation.csv` and `contact_sheet.png`.
CSV image paths are relative to the repository root; scores are on a 0–1 scale.
Inspect the contact sheet at full size and each original PNG before filling in
Yes/Partially/No, High/Medium/Low and notes. These fields are intentionally blank.
The script preserves an existing CSV to avoid overwriting manual reviews.
All generated review artifacts remain ignored by Git; retain them locally for
presentation alongside the Phase 7 images.

Functions in `prepare_visual_evaluation.py`:
- `load_results`: validate 15 unique completed pairs and resolve existing images.
- `save_template`: export measured data with blank manual fields, preserving reviews.
- `save_contact_sheet`: label all images with subject, file, prediction and score.
- `main`: create the review directory and run the exports.

Verification on 2026-10-18: all 31 tests passed; all 15 real classifier predictions
agreed with saved labels and scores, aggregates were recomputed, and local Tiny-SD
loaded successfully. Gradio interface construction succeeded, but fresh server
launches on 7860 and 7861 failed and localhost access returned Operation not
permitted even with requested escalation. Startup remains unverified in this
execution environment. Launch `app.py` locally for presentation rehearsal. No
new diffusion inference or browser upload/generation was performed in Phase 8.
