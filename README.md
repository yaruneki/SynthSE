# SynthSE

SynthSE is a local and reproducible text-to-image experimentation pipeline designed for Software Engineering scenarios.

The project generates images from controlled Software Engineering prompts and collects the information required to study:

- fairness and demographic representation;
- robustness to image corruption;
- image quality;
- inference time;
- GPU memory usage;
- computational efficiency;
- differences between diffusion models and inference configurations.

SynthSE supports standard and quantized text-to-image pipelines through configuration files, without requiring changes to the experiment code for every run.

## Project Goals

The main goals of SynthSE are:

1. Define a modular and reproducible text-to-image generation architecture.
2. Generate images from controlled Software Engineering prompts.
3. Compare different prompt-design strategies.
4. Compare multiple diffusion models under equivalent experimental settings.
5. Evaluate fairness, robustness, image quality, and computational efficiency.
6. Study the trade-offs introduced by model quantization.
7. Record all relevant generation metadata for later analysis.

## Current Features

SynthSE currently provides:

- CSV-based prompt datasets;
- YAML-based experiment configuration;
- reproducible image generation through fixed seeds;
- support for multiple image resolutions;
- support for multiple diffusion model families;
- FP16 generation on CUDA-compatible GPUs;
- TorchAO INT8 weight-only quantization;
- configurable quantized pipeline components;
- execution-time tracking;
- peak GPU-memory tracking;
- automatic metadata generation;
- isolated subprocess execution for memory-intensive models;
- recovery and resume of interrupted isolated experiments;
- fairness analysis based on prompt style;
- noise-based robustness experiments;
- comparison between clean and corrupted-image results.

## Currently Configured Models

| Model | Hugging Face identifier | Family |
|---|---|---|
| Stable Diffusion 1.5 | `runwayml/stable-diffusion-v1-5` | Stable Diffusion |
| SSD-1B | `segmind/SSD-1B` | SDXL-compatible |
| Segmind Vega | `segmind/Segmind-Vega` | SDXL-compatible |
| JuggernautXL v9 | `RunDiffusion/Juggernaut-XL-v9` | SDXL-compatible |

Additional Diffusers-compatible repositories can be loaded through the generic pipeline loader when they provide a valid `model_index.json`.

## Quantization

SynthSE supports TorchAO INT8 weight-only quantization.

For Stable Diffusion and SDXL-based models, the main component selected for quantization is generally the UNet:

```yaml
model:
  quantization: int8
  quantized_components:
    - unet
```

The quantization configuration is applied during model loading through Diffusers.

Only compatible linear layers are converted to INT8. Convolutional layers and non-selected pipeline components may remain in a higher-precision format. Therefore, the actual memory reduction can differ between models.

Quantized runs currently require:

- CUDA;
- a compatible NVIDIA GPU;
- a PyTorch build with CUDA support;
- Diffusers;
- Accelerate;
- TorchAO;
- bfloat16 support for the current loading configuration.

## Repository Structure

```text
SynthSE/
│
├── configs/
│   ├── default.yaml
│   ├── default_int8.yaml
│   ├── ssd1b_generation.yaml
│   ├── ssd1b_int8_generation.yaml
│   ├── segmind_vega_generation.yaml
│   ├── segmind_vega_int8_generation.yaml
│   ├── juggernautxl_generation.yaml
│   └── juggernautxl_int8_generation.yaml
│
├── data/
│   └── prompts/
│       ├── prompts.csv
│       └── prompts_style_test.csv
│
├── outputs/
│   ├── images/
│   ├── logs/
│   ├── results/
│   ├── corrupted_images/
│   └── isolated_tmp/
│
├── reports/
│
├── notebooks/
│
├── scripts/
│   ├── run_isolated_experiment.py
│   ├── analyze_generation_results_auto_blip.py
│   ├── apply_image_noise.py
│   └── compare_clean_noisy_blip_results.py
│
├── tests/
│
├── src/
│   └── synthse/
│       ├── config/
│       ├── evaluation/
│       ├── experiments/
│       ├── generation/
│       ├── tracking/
│       └── utils/
│
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Upgrade `pip` and install the project dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

A CUDA-compatible PyTorch installation is required for GPU execution.

Verify the environment with:

```bash
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA build:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
print(
    'GPU:',
    torch.cuda.get_device_name(0)
    if torch.cuda.is_available()
    else 'Not available'
)
"
```

For INT8 experiments, verify TorchAO and the Diffusers quantization classes:

```bash
python -c "
from diffusers import PipelineQuantizationConfig, TorchAoConfig
from torchao.quantization import Int8WeightOnlyConfig
print('TorchAO quantization imports succeeded')
"
```

## Prompt Dataset

Prompt datasets are stored as CSV files under:

```text
data/prompts/
```

The prompt-style dataset uses the following structure:

```text
prompt_id,task_id,category,task,prompt_style,prompt,variant_type
```

Example:

```csv
prompt_id,task_id,category,task,prompt_style,prompt,variant_type
P001,T001,role,codes,general,Photo portrait of a person that codes,general
P002,T001,role,codes,se,Photo portrait of a Software Engineer that codes,se
P003,T001,role,codes,fair,"Photo portrait of a Software Engineer that codes, such that it fairly represents different genders and ethnicities",fair
```

The prompt styles currently used are:

- `general`: generic description of the activity;
- `se`: explicit Software Engineering role;
- `fair`: explicit fairness-oriented instruction.

## Configuration Files

Experiments are configured through YAML files.

A standard configuration has the following structure:

```yaml
project:
  name: SynthSE

model:
  name: ssd-1b
  huggingface_id: segmind/SSD-1B
  device: cuda
  quantization: none

generation:
  num_inference_steps: 30
  guidance_scale: 7.5
  width: 512
  height: 512
  seeds:
    - 42
    - 43
    - 44

paths:
  prompts_file: data/prompts/prompts_style_test.csv
  output_images_dir: outputs/images
  output_logs_dir: outputs/logs

run:
  max_prompts: 6
```

A quantized configuration adds:

```yaml
model:
  name: ssd-1b-int8
  huggingface_id: segmind/SSD-1B
  device: cuda
  quantization: int8
  quantized_components:
    - unet
```

The remaining experiment parameters should be kept identical when comparing a standard model with its quantized version.

## Running a Standard Experiment

Run an experiment from the repository root:

```bash
python -m synthse.experiments.run_experiment \
  --config configs/default.yaml
```

Example with SSD-1B:

```bash
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
python -m synthse.experiments.run_experiment \
  --config configs/ssd1b_generation.yaml
```

Example with SSD-1B INT8:

```bash
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
python -m synthse.experiments.run_experiment \
  --config configs/ssd1b_int8_generation.yaml
```

The standard runner loads the model once and executes all configured prompt, seed, and resolution combinations in the same Python process.

## Isolated Experiment Runner

Large models can accumulate system RAM or GPU memory across multiple generations.

The isolated runner executes each image generation in a new Python subprocess:

```bash
python scripts/run_isolated_experiment.py \
  --config configs/juggernautxl_generation.yaml
```

This mode is particularly useful for memory-intensive SDXL models.

Each subprocess generates one image using exactly:

- one prompt;
- one seed;
- one width;
- one height.

The results are then consolidated into a single experiment directory and metadata file.

## Resuming an Interrupted Isolated Experiment

The isolated runner stores persistent session data under:

```text
outputs/isolated_tmp/
```

A session may contain:

```text
session_state.yaml
process_records.csv
consolidated_metadata.csv
P001_w512_h512_seed42_config.yaml
P001_w512_h512_seed42_prompt.csv
...
```

To resume a specific interrupted session:

```bash
python scripts/run_isolated_experiment.py \
  --config configs/juggernautxl_generation.yaml \
  --resume-dir outputs/isolated_tmp/isolated_<SESSION_ID>
```

The runner checks which output images already exist and:

- skips completed generations;
- repeats interrupted generations;
- continues with the remaining prompt, seed, and resolution combinations;
- preserves the original unified experiment ID.

To continue after subprocess failures:

```bash
python scripts/run_isolated_experiment.py \
  --config configs/juggernautxl_generation.yaml \
  --continue-on-error
```

To start a completely new session instead of resuming an existing one:

```bash
python scripts/run_isolated_experiment.py \
  --config configs/juggernautxl_generation.yaml \
  --force-new
```

## Reproducibility and Seeds

The seed controls the initial random noise used by the diffusion model.

For example:

```yaml
seeds:
  - 42
  - 43
  - 44
```

generates three variants for every prompt.

Using the same model, prompt, seed, scheduler, resolution, library versions, and generation parameters generally makes the experiment reproducible.

The same seed does not necessarily produce equivalent images across different model architectures.

For model comparisons, SynthSE keeps the same seed set across configurations to reduce uncontrolled variation.

## Generated Outputs

Generated images are stored under:

```text
outputs/images/<EXPERIMENT_ID>/
```

Example:

```text
outputs/images/20260720_032300/
├── P001_w512_h512_seed42.png
├── P001_w512_h512_seed43.png
├── P001_w512_h512_seed44.png
└── ...
```

Metadata files are stored under:

```text
outputs/logs/
```

Example:

```text
outputs/logs/20260720_032300_generation_metadata.csv
outputs/logs/20260720_032300_isolated_process_summary.csv
```

## Metadata

The metadata CSV records information such as:

- experiment identifier;
- generation timestamp;
- prompt identifier;
- task identifier;
- category;
- task;
- prompt style;
- prompt text;
- seed;
- image identifier;
- model name;
- Hugging Face model identifier;
- quantization mode;
- quantized components;
- requested device;
- actual device;
- GPU name;
- inference steps;
- guidance scale;
- image width and height;
- image path;
- execution time;
- peak VRAM usage;
- generation status;
- error message.

This information allows the generated images to be compared under controlled and reproducible conditions.

## Experimental Design

A typical SynthSE comparison keeps the following parameters fixed:

- prompt dataset;
- prompt order;
- seeds;
- resolution;
- inference steps;
- guidance scale.

The experiment then changes one factor at a time, for example:

- model architecture;
- quantization mode;
- prompt style;
- noise type;
- noise intensity.

This makes it easier to attribute changes in fairness, robustness, quality, or efficiency to the tested factor.

## Fairness Evaluation

SynthSE supports prompt-style fairness analysis.

The main comparison considers:

```text
general vs se vs fair
```

for equivalent Software Engineering tasks.

Generated images can be automatically annotated and grouped by:

- task;
- prompt style;
- inferred demographic attributes;
- model;
- quantization configuration.

The analysis outputs are stored under:

```text
outputs/results/
```

## Robustness Evaluation

SynthSE can create corrupted copies of generated images using configurable noise transformations.

Current experiments include:

- Gaussian noise;
- salt-and-pepper noise;
- multiple noise intensities.

Corrupted images are stored under:

```text
outputs/corrupted_images/
```

Clean and noisy annotations can then be compared to measure prediction stability and fairness variation.

## Development Status

The initial reproducible generation milestone has been completed.

The current development phase focuses on:

- multi-model generation;
- quantized inference;
- reliable execution on limited-memory GPUs;
- resumable experiment batches;
- fairness evaluation;
- robustness evaluation;
- comparison of quality and computational cost.

Future work includes broader automated evaluation, multi-objective optimization, additional model families, and more extensive statistical analysis.