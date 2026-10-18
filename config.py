"""Central configuration for StyleLoop."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Phase 2: local dataset preparation and ResNet18 input preprocessing.
RAW_DATA_DIR = DATA_DIR / "raw"
PREPARED_DATA_DIR = DATA_DIR / "prepared"
STYLE_FOLDERS = {
    "Impressionism": "Impressionism",
    "Cubism": "Cubism",
    "Expressionism": "Expressionism",
    "Abstract Expressionism": "Abstract_Expressionism",
    "Pop Art": "Pop_Art",
}
IMAGES_PER_CLASS = 200
RANDOM_SEED = 42
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
IMAGE_SIZE = 224
HORIZONTAL_FLIP_PROBABILITY = 0.5
NORMALIZATION_MEAN = (0.485, 0.456, 0.406)
NORMALIZATION_STD = (0.229, 0.224, 0.225)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Phase 3: frozen-backbone transfer learning.
EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_WORKERS = 0
CPU_THREADS = 4
CHECKPOINT_PATH = MODELS_DIR / "style_classifier.pt"
WEIGHTS_CACHE_DIR = MODELS_DIR / "pretrained"
PRETRAINED_WEIGHTS_PATH = WEIGHTS_CACHE_DIR / "checkpoints" / "resnet18-f37072fd.pth"

# Local CPU generation; Phase 6.5 defaults balance visual quality and latency.
GENERATOR_MODEL_ID = "segmind/tiny-sd"
GENERATOR_MODEL_DIR = MODELS_DIR / "pretrained" / "tiny-sd"
GENERATED_DIR = OUTPUTS_DIR / "generated"
GENERATOR_DEVICE = "cpu"
GENERATOR_WIDTH = 512
GENERATOR_HEIGHT = 512
GENERATOR_STEPS = 20
GENERATOR_GUIDANCE_SCALE = 7.5
GENERATOR_SEED = 42
STYLE_DESCRIPTIONS = {
    "Impressionism": "Impressionist oil painting, visible brushstrokes, soft natural light, atmospheric colors",
    "Cubism": "Cubist painting, geometric shapes, fragmented forms, multiple perspectives",
    "Expressionism": "Expressionist painting, expressive brushwork, distorted forms, intense emotional colors",
    "Abstract Expressionism": "Abstract Expressionist painting, gestural brushstrokes, spontaneous marks, nonrepresentational color fields",
    "Pop Art": "Pop Art painting, bold flat colors, strong outlines, graphic shapes, comic-book imagery",
}
EXPERIMENTS_DIR = OUTPUTS_DIR / "experiments"
EXPERIMENT_SUBJECT = "A woman sitting on a bench in a colorful flower garden"
EXPERIMENT_STYLE = "Impressionism"
EXPERIMENT_CONFIGURATIONS = [("A", 256, 256, 10), ("B", 512, 512, 20), ("C", 512, 512, 30)]
GENERATOR_TEST_SUBJECT = "a small house near a lake"
GENERATOR_TEST_STYLE = "Impressionism"
STYLELOOP_DIR = OUTPUTS_DIR / "styleloop"
STYLELOOP_DEMO_SUBJECT = "a small house near a lake"
GRADIO_HOST = "127.0.0.1"
GRADIO_PORT = 7860
GENERATOR_MIN_FREE_BYTES = 5 * 1024**3
# Select the original .bin weights, excluding alternate formats and variants.
GENERATOR_DOWNLOAD_PATTERNS = [
    "model_index.json", "README.md", "LICENSE*", "scheduler/*.json",
    "tokenizer/*", "text_encoder/config.json", "text_encoder/pytorch_model.bin",
    "unet/config.json", "unet/diffusion_pytorch_model.bin",
    "vae/config.json", "vae/diffusion_pytorch_model.bin",
    "feature_extractor/*.json",
]

# Phase 7: one generation per style/subject, paired with distinct correct inputs.
PHASE7_DIR = EXPERIMENTS_DIR / "phase7"
PHASE7_SUBJECTS = (
    "A woman sitting in a flower garden",
    "A street in a European city at sunset",
    "A small sailing boat on the sea",
)

# Phase 8: manual review artifacts derived only from the saved Phase 7 experiment.
PHASE8_DIR = EXPERIMENTS_DIR / "phase8"
