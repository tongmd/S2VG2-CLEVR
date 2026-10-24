# S2VG2: Human-Readable SVG Generation and Semantic Visual Reasoning

[Latest manuscript (PDF)](https://beyondpixels.stonezhang.com/paper.pdf)

Related paper: [Project page](https://beyondpixels.stonezhang.com/) | [Paper (arXiv)](https://arxiv.org/abs/2311.15543)

This repository contains code, generated datasets, and inference results for evaluating explicit Vision-Language Models on **semantic SVG** representations.

Our core claim is simple: when an image is converted into a compact, human-readable SVG scene graph, frontier language models can reason over the scene directly from text.

## Contents

- **`generate_dataset.py`**: Generates the CLEVR-SVG benchmark and runs zero-shot reasoning evaluation.
- **`CLEVR_SVG_Dataset.json` / `CLEVR_SVG_Dataset_200.json`**: Procedurally generated benchmark samples.
- **`CLEVR_SVG_Results.json` / `CLEVR_SVG_Results_100.json`**: Raw model outputs and evaluation logs.

## Setup

This repo does **not** contain any API keys.

To rerun the evaluation, set your Gemini API key in the environment:

```bash
export GEMINI_API_KEY="your_key_here"
python generate_dataset.py
```

## Experimental Summary

In our zero-shot setting, the model reads only the SVG text, not the raster image. This yields strong performance on counting, spatial relation, and attribute-query tasks, supporting the paper's argument that semantic SVG acts as an effective structured intermediate representation for visual reasoning.

## Citation

Please cite the corresponding paper / arXiv version when available.
