# SynthSE resource metrics analysis

## Experiment 20260723_195347

- Model: `runwayml/stable-diffusion-v1-5`
- Quantization: `int8`
- Device: `cuda`
- GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`
- Successful images: 36
- Failed images: 0
- Mean generation time: 5.4674 s
- Median generation time: 5.4462 s
- Mean GPU energy per generated image: 189.2712 J
- Total generation GPU energy: 6813.7620 J
- Total model-load GPU energy: 66.4000 J
- End-to-end GPU energy: 6880.1620 J
- End-to-end GPU energy per image: 191.1156 J
- Mean GPU power: 32.1460 W
- Mean GPU utilization: 84.2200%
- Peak NVML VRAM: 3578.6172 MiB
- End-to-end throughput: 634.3514 images/hour

## Warm-up observations

- 20260723_195347 / generation_total_seconds: warm-up mean 6.3080, steady-state mean 5.4434, difference 15.8841%.
- 20260723_195347 / gpu_energy_j: warm-up mean 202.8930, steady-state mean 188.8820, difference 7.4179%.
