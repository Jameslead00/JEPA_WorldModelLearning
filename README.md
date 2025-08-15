# JEPA World Model Learning – Frozen Lake AI Agent

This project implements a Joint Embedding Predictive Architecture (JEPA) world model to solve the Frozen Lake environment from OpenAI Gymnasium. It demonstrates advanced AI engineering skills in model-based reinforcement learning, environment simulation, and agent planning.

## What I Did

- **Developed a JEPA World Model:** Built a modular JEPA architecture with custom encoder, predictor, and decoder networks for latent state modeling.
- **Trained and Evaluated the Model:** Used PyTorch Lightning for training; loaded and tested a trained checkpoint to validate agent performance.
- **Implemented Dreaming and Dream Search:** Enabled the agent to "dream" future states and plan optimal paths using breadth-first search (BFS) in latent space.
- **Integrated Visualization:** Converted latent states to images for debugging and analysis; visualized agent trajectories and predicted outcomes.
- **Human Play and Environment Rendering:** Added support for human interaction and text/graphical rendering of the Frozen Lake environment.
- **Project Structure & Automation:** Organized code for reproducibility and scalability; managed dependencies with `uv` and `pyproject.toml`.

## Highlight Technologies

- **PyTorch & PyTorch Lightning:** Deep learning framework for model definition, training, and checkpointing.
- **OpenAI Gymnasium:** Standardized RL environment for agent simulation.
- **NumPy & Matplotlib:** Data manipulation and visualization.
- **Scikit-learn:** Utility functions for analysis.
- **Pygame:** Interactive environment rendering.
- **Breadth-First Search (BFS):** Systematic planning in latent space.
- **uv:** Modern Python environment and dependency manager.

## Key Files

- [`train.py`](train.py): Model training script.
- [`eval.py`](eval.py): Evaluation, dreaming, and planning logic.
- [`frozen_lake.py`](frozen_lake.py): Custom Frozen Lake environment implementation.
- [`game.py`](game.py): Game loop and agent interaction.
- [`human_play.py`](human_play.py): Human agent interface.
- [`dataset.py`](dataset.py): Dataset preparation and loading.
- [`jepa_model_vicreg.ckpt`](jepa_model_vicreg.ckpt): Trained model checkpoint.

## How to Run

Install dependencies and run any script using:

```bash
uv run python <script.py>
```
