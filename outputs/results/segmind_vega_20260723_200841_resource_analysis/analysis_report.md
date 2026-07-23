# SynthSE resource metrics analysis

## Experiment 20260723_200841

- Model: `segmind/Segmind-Vega`
- Quantization: `none`
- Device: `cuda`
- GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`
- Successful images: 36
- Failed images: 0
- Mean generation time: 4.5273 s
- Median generation time: 4.3870 s
- Mean GPU energy per generated image: 123.7316 J
- Total generation GPU energy: 4454.3380 J
- Total model-load GPU energy: 72.1170 J
- End-to-end GPU energy: 4526.4550 J
- End-to-end GPU energy per image: 125.7349 J
- Mean GPU power: 25.0850 W
- Mean GPU utilization: 69.2470%
- Peak NVML VRAM: 2302.6172 MiB
- End-to-end throughput: 754.2448 images/hour

## Warm-up observations

- 20260723_200841 / generation_total_seconds: warm-up mean 7.1608, steady-state mean 4.4521, difference 60.8399%.
- 20260723_200841 / gpu_energy_j: warm-up mean 149.2870, steady-state mean 123.0015, difference 21.3701%.
