# vllm_fine_tune — Qwen3 VL (8B) Vision examples

Minimal project containing a notebook and a runnable script for Qwen3 Vision (8B) experiments.

## Contents
- `qwen3/Qwen3_VL_(8B)_Vision.ipynb` — Interactive notebook for experimentation.
- `qwen3/main.py` — Main script to run the modularized pipeline.
- `qwen3/model_setup.py` — Contains functions to load and configure the model.
- `qwen3/data_processing.py` — Handles dataset loading and preprocessing.
- `qwen3/training.py` — Defines the training pipeline for the model.
- `qwen3/inference.py` — Contains the inference logic for the model.
- `qwen3/outputs/` — Directory for storing training outputs.
- `qwen3/latex_output.txt` — File containing LaTeX representation of a dataset sample.

## Dataset
This project uses the `unsloth/LaTeX_OCR` dataset, which contains images of handwritten mathematical formulas paired with their LaTeX representations. The dataset is ideal for training vision-language models to convert images of mathematical content into machine-readable LaTeX code. The dataset is loaded using the `datasets` library.

## Application
The code demonstrates the following applications:
1. **Fine-tuning Vision-Language Models**: The Qwen3 VL (8B) model is fine-tuned on the `unsloth/LaTeX_OCR` dataset to improve its ability to interpret and convert mathematical images into LaTeX.
2. **Inference**: The fine-tuned model can be used to generate LaTeX code for new images of mathematical formulas.
3. **Efficient Training with LoRA**: The project uses Low-Rank Adaptation (LoRA) to fine-tune the model efficiently, reducing memory usage and computational cost.

## Requirements
Install dependencies (Linux):
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage
Run the main script:
```bash
python3 main.py
```

## Description
This project demonstrates fine-tuning and inference for the Qwen3 VL (8B) Vision model using the Unsloth library. The code is modularized into separate files for better readability and maintainability.

