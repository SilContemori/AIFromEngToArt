"""Local Tiny-SD generation; imports do not load weights or contact the network."""

from functools import lru_cache
from uuid import uuid4

import config


def build_generation_prompt(subject, style):
    """Combine the subject with the predicted style's visual characteristics."""
    if not isinstance(subject, str) or not subject.strip():
        raise ValueError("Subject must be a non-empty string")
    if not isinstance(style, str) or style.strip() not in config.STYLE_DESCRIPTIONS:
        raise ValueError(f"Unknown artistic style: {style}")
    return f"{subject.strip()}, {config.STYLE_DESCRIPTIONS[style.strip()]}"


@lru_cache(maxsize=1)
def load_generator():
    """Load and cache the local Diffusers pipeline on CPU in float32."""
    if not (config.GENERATOR_MODEL_DIR / "model_index.json").is_file():
        raise FileNotFoundError(f"Download Tiny-SD first: {config.GENERATOR_MODEL_DIR}")
    import torch
    from diffusers import StableDiffusionPipeline

    torch.set_num_threads(config.CPU_THREADS)
    pipeline = StableDiffusionPipeline.from_pretrained(
        str(config.GENERATOR_MODEL_DIR), local_files_only=True,
        torch_dtype=torch.float32, use_safetensors=False,
        low_cpu_mem_usage=False,
    )
    return pipeline.to(config.GENERATOR_DEVICE)


def generate_image(subject, style, pipeline, *, prompt=None, output_path=None,
                   width=None, height=None, steps=None, guidance_scale=None, seed=None):
    """Generate, save one PNG with a unique name, and return its PIL image."""
    if not isinstance(subject, str) or not subject.strip():
        raise ValueError("Subject must be a non-empty string")
    if not isinstance(style, str) or not style.strip():
        raise ValueError("Style must be a non-empty string")
    width = config.GENERATOR_WIDTH if width is None else width
    height = config.GENERATOR_HEIGHT if height is None else height
    steps = config.GENERATOR_STEPS if steps is None else steps
    guidance_scale = config.GENERATOR_GUIDANCE_SCALE if guidance_scale is None else guidance_scale
    seed = config.GENERATOR_SEED if seed is None else seed
    for name, value in (("width", width), ("height", height)):
        if type(value) is not int or not 64 <= value <= 1024 or value % 8:
            raise ValueError(f"{name} must be a multiple of 8 between 64 and 1024")
    if type(steps) is not int or not 1 <= steps <= 100:
        raise ValueError("Steps must be an integer between 1 and 100")
    import math
    if not isinstance(guidance_scale, (int, float)) or not math.isfinite(guidance_scale) or not 1 <= guidance_scale <= 30:
        raise ValueError("Guidance scale must be between 1 and 30")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("Seed must be an integer between 0 and 2**63 - 1")
    import torch

    if prompt is None:
        prompt = build_generation_prompt(subject, style)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt must be a non-empty string")
    with torch.inference_mode():
        result = pipeline(
            prompt=prompt, width=width, height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=1,
            generator=torch.Generator(device=config.GENERATOR_DEVICE).manual_seed(
                seed
            ),
        )
    image = result.images[0]
    config.GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = config.GENERATED_DIR / f"tiny-sd-{uuid4().hex}.png"
    image.save(output_path)
    print(f"Saved image: {output_path}")
    return image


if __name__ == "__main__":
    import json
    import resource
    import sys
    from time import perf_counter

    start = perf_counter()
    pipeline = load_generator()
    loaded = perf_counter()
    image = generate_image(config.GENERATOR_TEST_SUBJECT, config.GENERATOR_TEST_STYLE, pipeline)
    finished = perf_counter()
    # ru_maxrss is bytes on macOS and KiB on Linux; it includes model loading.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mib = peak / (1024**2 if sys.platform == "darwin" else 1024)
    report = {
        "model": config.GENERATOR_MODEL_ID, "device": config.GENERATOR_DEVICE,
        "seed": config.GENERATOR_SEED, "steps": config.GENERATOR_STEPS,
        "image_dimensions": list(image.size), "load_seconds": loaded - start,
        "generation_and_save_seconds": finished - loaded,
        "total_seconds": finished - start, "process_peak_rss_mib": peak_mib,
    }
    print(json.dumps(report, indent=2))
    (config.GENERATED_DIR / "last_run.json").write_text(json.dumps(report, indent=2) + "\n")
