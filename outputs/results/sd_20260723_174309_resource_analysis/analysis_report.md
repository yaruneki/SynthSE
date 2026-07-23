# SynthSE resource metrics analysis

## Experiment 20260723_174309

- Model: `runwayml/stable-diffusion-v1-5`
- Quantization: `none`
- Device: `cuda`
- GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`
- Successful images: 36
- Failed images: 0
- Mean generation time: 6.5292 s
- Median generation time: 6.4228 s
- Mean GPU energy per generated image: 196.4113 J
- Total generation GPU energy: 7070.8050 J
- Total model-load GPU energy: 66.5600 J
- End-to-end GPU energy: 7137.3650 J
- End-to-end GPU energy per image: 198.2601 J
- Mean GPU power: 28.3246 W
- Mean GPU utilization: 74.9666%
- Peak NVML VRAM: 3088.6172 MiB
- End-to-end throughput: 533.4231 images/hour

## Warm-up observations

- 20260723_174309 / generation_total_seconds: warm-up mean 9.4187, steady-state mean 6.4466, difference 46.1031%.
- 20260723_174309 / gpu_energy_j: warm-up mean 218.4540, steady-state mean 195.7815, difference 11.5805%.
