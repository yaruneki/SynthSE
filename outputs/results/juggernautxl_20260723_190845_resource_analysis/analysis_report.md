# SynthSE resource metrics analysis

## Experiment 20260723_190845

- Model: `RunDiffusion/Juggernaut-XL-v9`
- Quantization: `none`
- Device: `cuda`
- GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`
- Successful images: 36
- Failed images: 0
- Mean generation time: 20.5011 s
- Median generation time: 19.1363 s
- Mean GPU energy per generated image: 281.1191 J
- Total generation GPU energy: 10120.2880 J
- Total model-load GPU energy: 2709.2790 J
- End-to-end GPU energy: 12829.5670 J
- End-to-end GPU energy per image: 356.3769 J
- Mean GPU power: 13.4435 W
- Mean GPU utilization: 34.3645%
- Peak NVML VRAM: 5710.6172 MiB
- End-to-end throughput: 125.3027 images/hour

## Warm-up observations

- 20260723_190845 / generation_total_seconds: warm-up mean 29.8182, steady-state mean 20.2349, difference 47.3605%.
- 20260723_190845 / gpu_energy_j: warm-up mean 315.5910, steady-state mean 280.1342, difference 12.6571%.
