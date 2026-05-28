# SynthSE

SynthSE is a local text-to-image generation pipeline for Software Engineering scenarios.

The project aims to generate images from Software Engineering-related prompts and evaluate them with respect to fairness, robustness, image quality, and computational efficiency.

## Goals

- Define a modular and reproducible text-to-image generation architecture.
- Generate images from controlled Software Engineering prompts.
- Log prompts, seeds, generation parameters, execution time, hardware information, generated images, and metadata.
- Evaluate generated images with respect to fairness, robustness, image quality, and computational efficiency.

## Repository Structure

```text
configs/        Configuration files
data/prompts/   Prompt datasets and test cases
src/synthse/    Source code of the system
outputs/        Generated images, logs, and raw results
notebooks/      Exploratory analysis notebooks
reports/        Final report material
tests/          Unit and integration tests
```

## First Milestone

The first milestone is to implement a minimal reproducible generation pipeline:

1. Load prompts from CSV.
2. Load generation configuration from YAML.
3. Generate images locally.
4. Save generated images.
5. Log metadata for each generated image.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Run

```bash
python -m synthse.experiments.run_experiment --config configs/default.yaml
```

## Repository Structure

```text
SynthSE/
│
├── configs/              Configuration files
│   └── default.yaml
│
├── data/
│   └── prompts/          Prompt datasets and test cases
│       └── prompts.csv
│
├── outputs/              Generated artifacts
│   ├── images/           Generated images
│   ├── logs/             Metadata logs
│   └── results/          Raw experimental results
│
├── reports/              Report and documentation material
├── notebooks/            Exploratory analysis notebooks
├── scripts/              Utility scripts
├── tests/                Unit and integration tests
│
└── src/
    └── synthse/          Source code of the system
        ├── config/       Configuration loading
        ├── generation/   Text-to-image generation logic
        ├── experiments/  Experiment runner
        ├── evaluation/   Fairness, quality, robustness and efficiency metrics
        ├── tracking/     Metadata logging
        └── utils/        Utility functions
```
