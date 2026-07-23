# SynthSE resource metrics analysis

## Experiment 20260723_224105

- Model: `segmind/Segmind-Vega`
- Quantization: `int8`
- Device: `cuda`
- GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`
- Successful images: 36
- Failed images: 0
- Mean generation time: 2.7227 s
- Median generation time: 2.6796 s
- Mean GPU energy per generated image: 114.0596 J
- Total generation GPU energy: 4106.1460 J
- Total model-load GPU energy: 152.2400 J
- End-to-end GPU energy: 4258.3860 J
- End-to-end GPU energy per image: 118.2885 J
- Mean GPU power: 33.9433 W
- Mean GPU utilization: 67.8765%
- Peak NVML VRAM: 4212.6172 MiB
- End-to-end throughput: 1145.0449 images/hour

## Warm-up observations

- 20260723_224105 / generation_total_seconds: warm-up mean 4.8129, steady-state mean 2.6629, difference 80.7370%.
- 20260723_224105 / gpu_energy_j: warm-up mean 123.1070, steady-state mean 113.8011, difference 8.1773%.
