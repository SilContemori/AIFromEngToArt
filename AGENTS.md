# StyleLoop — Development Guidelines

## Project Goal
Develop a simple university project for the course "AI From Engineering to Arts".

The application must:
1. Classify the artistic style of an input image using a pretrained CNN with transfer learning.
2. Generate a new image using a pretrained diffusion model, conditioned on the predicted artistic style and a user-provided subject.
3. Classify the generated image again to evaluate style consistency.
4. Provide a simple Gradio interface.

## Coding Rules
- Use Python.
- Use uv for environment and dependency management, with dependencies in `pyproject.toml` and the lockfile in `uv.lock`.
- Keep the implementation simple, readable and suitable for a university project.
- Prefer functions over classes unless a class is strictly necessary.
- Use descriptive variable names and short functions.
- Add short comments explaining non-obvious logic.
- Avoid unnecessary abstractions, design patterns and complex architecture.
- Avoid databases, APIs, Docker and external services unless strictly necessary.
- Introduce dependencies only when needed for the requested phase; prefer the standard library for setup and tests.
- For later ML phases, use widely adopted libraries: PyTorch, torchvision, scikit-learn, diffusers, Gradio, pandas and matplotlib.
- Keep all configurable parameters in a single configuration file.
- Do not implement neural network architectures from scratch when pretrained models are appropriate.
- Do not add new features without explicit authorization.

## Workflow Rules
- Implement only the requested development phase.
- Explain which files were created or modified.
- Explain the purpose of every new function.
- Run simple tests whenever possible.
- Never claim that training or inference succeeded unless actually executed.
- Before introducing additional dependencies or complexity, explain why they are needed.
- After each phase, provide instructions to run and verify the result.

## Documentation
- Maintain a simple README with setup and execution instructions.
- Use English for code, variable names and documentation.
- Keep the project easy for a student to navigate, understand and explain.

## Current Phase Boundary
- Phase 1 contains only project setup, directory configuration, module placeholders, and smoke tests.
- Do not implement dataset processing, CNN classification, training, diffusion generation, pipeline evaluation, or Gradio until the corresponding phase is explicitly requested.
- Placeholder modules must import without third-party dependencies or side effects.
