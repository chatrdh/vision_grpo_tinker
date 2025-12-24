# Visual R1 GRPO Training

Training Vision Language Models (VLMs) on visual math reasoning tasks using Group Relative Policy Optimization (GRPO) with the [Tinker](https://tinker.build) training API.

## Overview

This project fine-tunes **Qwen/Qwen3-VL-30B-A3B-Instruct** on the [Geometry3K](https://huggingface.co/datasets/hiyouga/geometry3k) dataset using GRPO.

### Key Features
- **GRPO Training**: Group-normalized advantage computation for stable RL training
- **PPO-Style Clipping**: Optional clipped importance sampling loss (`train_clipped_grpo.py`)
- **Chain-of-Thought**: Trains models to reason step-by-step with `<think>...</think>` tags
- **Visual Grounding Rewards**: Bonus rewards for referencing visual elements
- **Tinker Integration**: Leverages Tinker's distributed training infrastructure

## Project Structure

```
vision_grpo_tinker/
├── train.py                  # Main GRPO training (importance sampling loss)
├── train_clipped_grpo.py     # GRPO with PPO-style clipped loss
├── eval.py                   # Evaluation script with wandb logging
├── dataset.py                # Geometry3K dataset loader
├── reward.py                 # Reward functions (format, accuracy, visual grounding)
├── grpo_utils.py             # GRPO advantage computation
└── grader.py                 # Answer grading utilities
```

## Setup

```bash
# Install dependencies
pip install tinker transformers torch pandas pillow tqdm wandb

# Set Tinker API key
export TINKER_API_KEY="your-api-key"

# Download dataset (automatic on first run)
```

## Training

### Standard GRPO (Importance Sampling Loss)
```bash
python train.py
```

### GRPO with Clipped Loss
```bash
python train_clipped_grpo.py
```

### Resume Training
Set `resume_from_step` in the CONFIG dict to resume from a checkpoint:
```python
CONFIG = {
    "resume_from_step": 350,  # Or None to start fresh
    ...
}
```

## Evaluation

```bash
python eval.py
```

Results are logged to Weights & Biases with:
- Accuracy, format rate, visual grounding rate
- Predictions table with sample outputs

## Configuration

Key hyperparameters in `CONFIG`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `group_size` | 4 | Number of samples per group for GRPO |
| `batch_size` | 4 | Batch size |
| `max_steps` | 500 | Total training steps |
| `learning_rate` | 1e-6 | Learning rate |
| `temperature` | 0.9 | Sampling temperature |
| `clip_epsilon` | 0.2 | PPO clipping (clipped version only) |
| `lora_rank` | 16 | LoRA adapter rank |

## Reward Structure

### Standard Rewards (Default)
- **Format Reward** (0.5): Proper `<think>` and `<answer>` tags
- **Accuracy Reward** (1.0): Correct answer
- **Visual Grounding** (0.1): References to visual elements (angles, shapes, etc.)

### Vision-SR1 "Blindfolded Test" (Alternative)

A dual-pass reward system inspired by Vision-SR1 that encourages genuine visual understanding:

**Key Insight**: If a model truly understands an image, it should describe it well enough that it can solve the problem *without* looking at the image again.

| Reward Component | Value | Description |
|------------------|-------|-------------|
| Format (full) | 0.1 | Proper `<scan>` + `<think>` + `<answer>` tags |
| Accuracy (Pass 1) | 1.0 | Correct answer with image |
| Blind Correct | +1.0 | Pass 2 correct → scan was sufficient |
| Blind Wrong | -0.5 | Pass 2 wrong → scan was insufficient |

**How it works**:
1. **Pass 1 (with image)**: Model generates `<scan>` (visual description) + `<think>` + `<answer>`
2. **Pass 2 (blind)**: Model solves using *only* the question + `<scan>` (no image)
3. If Pass 2 succeeds, the visual description was sufficient → bonus reward

Use `vision_sr1_reward_fn()` in `reward.py` for this approach.

## Checkpoints

Training checkpoints are saved every 50 steps with:
- `save_state()`: Full state (model + optimizer) for resuming training
- `save_weights_for_sampler()`: Inference-optimized weights for evaluation
