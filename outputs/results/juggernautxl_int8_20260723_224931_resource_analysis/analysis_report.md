# SynthSE resource metrics analysis

## Experiment 20260723_224931

- Model: `RunDiffusion/Juggernaut-XL-v9`
- Quantization: `int8`
- Device: `cuda`
- GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`
- Successful images: 36
- Failed images: 0
- Mean generation time: 4.9009 s
- Median generation time: 4.8485 s
- Mean GPU energy per generated image: 212.5758 J
- Total generation GPU energy: 7652.7280 J
- Total model-load GPU energy: 121.7620 J
- End-to-end GPU energy: 7774.4900 J
- End-to-end GPU energy per image: 215.9581 J
- Mean GPU power: 34.5080 W
- Mean GPU utilization: 66.7040%
- Peak NVML VRAM: 6080.6172 MiB
- End-to-end throughput: 681.3334 images/hour

## Warm-up observations

- 20260723_224931 / generation_total_seconds: warm-up mean 6.5969, steady-state mean 4.8525, difference 35.9502%.
- 20260723_224931 / gpu_energy_j: warm-up mean 206.8390, steady-state mean 212.7397, difference -2.7737%.
