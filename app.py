"""Simple local Gradio interface for the existing StyleLoop pipeline."""

import logging

import config
from pipeline import run_styleloop


def interface_error(message):
    """Create a Gradio error only when the callback needs to display one."""
    import gradio as gr
    return gr.Error(message)


def generate_artwork(input_image_path, subject, resolution=None, steps=None,
                     guidance_scale=None, seed=None):
    """Validate UI inputs, run StyleLoop, and format its eight display outputs."""
    if not input_image_path:
        raise interface_error("Please upload an artwork image.")
    if not isinstance(subject, str) or not subject.strip():
        raise interface_error("Please enter a subject to generate.")
    try:
        settings = {}
        if resolution is not None:
            settings.update(width=resolution, height=resolution)
        for name, value in (("steps", steps), ("guidance_scale", guidance_scale), ("seed", seed)):
            if value is not None:
                settings[name] = value
        result = run_styleloop(input_image_path, subject, **settings)
    except (ValueError, OSError) as error:
        raise interface_error(f"Could not process the artwork: {error}") from error
    except Exception as error:
        logging.exception("StyleLoop generation failed")
        raise interface_error("Generation failed. Check the terminal for details and try again.") from error
    return (
        result["input_image_path"],
        result["original_predicted_style"],
        result["generated_prompt"],
        result["generated_image_path"],
        result["generated_predicted_style"],
        f'{result["style_consistency_score"] * 100:.2f}%',
        "Yes" if result["style_match"] else "No",
        f'{result["execution_time_seconds"]:.2f} seconds',
    )


def build_interface():
    """Build two columns and connect the generate button to the pipeline callback."""
    import gradio as gr

    with gr.Blocks(title="StyleLoop") as demo:
        gr.Markdown("# StyleLoop\n### From an artwork's style to a new subject")
        gr.Markdown("Upload a painting and describe what you would like to create.")
        with gr.Row():
            with gr.Column():
                artwork = gr.Image(label="Original artwork", type="filepath", sources=["upload"])
                subject = gr.Textbox(label="Subject to generate", placeholder=config.STYLELOOP_DEMO_SUBJECT)
                with gr.Accordion("Generation settings", open=False):
                    resolution = gr.Dropdown(
                        choices=[256, 512], value=config.GENERATOR_WIDTH, label="Square image size (pixels)"
                    )
                    steps = gr.Slider(1, 100, value=config.GENERATOR_STEPS, step=1, label="Inference steps")
                    guidance = gr.Slider(1, 30, value=config.GENERATOR_GUIDANCE_SCALE, step=0.5, label="Guidance scale")
                    seed = gr.Number(value=config.GENERATOR_SEED, precision=0, label="Random seed")
                generate = gr.Button("Generate Artwork", variant="primary")
                original_style = gr.Textbox(label="Predicted input style", interactive=False)
                prompt = gr.Textbox(label="Generated prompt", lines=3, interactive=False)
            with gr.Column():
                generated = gr.Image(label="Generated artwork", interactive=False)
                generated_style = gr.Textbox(label="Predicted generated style", interactive=False)
                with gr.Row():
                    consistency = gr.Textbox(label="Style consistency score", interactive=False)
                    match = gr.Textbox(label="Styles match", interactive=False)
                duration = gr.Textbox(label="Pipeline execution time", interactive=False)
                gr.Markdown(
                    "The score is a classifier-based proxy for stylistic consistency, "
                    "not an objective measure of artistic quality or a calibrated "
                    "probability of correctness."
                )
        # One request at a time shares the cached models without concurrent inference.
        generate.click(
            generate_artwork, inputs=[artwork, subject, resolution, steps, guidance, seed],
            outputs=[artwork, original_style, prompt, generated, generated_style,
                     consistency, match, duration],
            concurrency_limit=1,
        )
    return demo.queue(default_concurrency_limit=1)


def main():
    """Launch only on localhost, without a public sharing link."""
    build_interface().launch(
        server_name=config.GRADIO_HOST, server_port=config.GRADIO_PORT,
        share=False, inbrowser=False,
    )


if __name__ == "__main__":
    main()
