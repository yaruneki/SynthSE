# SynthSE resource metrics analysis

## Experiment 20260723_200342

- Model: `segmind/SSD-1B`
- Quantization: `int8`
- Device: `cuda`
- GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`
- Successful images: 36
- Failed images: 0
- Mean generation time: 4.3099 s
- Median generation time: 4.2452 s
- Mean GPU energy per generated image: 150.1148 J
- Total generation GPU energy: 5404.1330 J
- Total model-load GPU energy: 83.1660 J
- End-to-end GPU energy: 5487.2990 J
- End-to-end GPU energy per image: 152.4250 J
- Mean GPU power: 31.7705 W
- Mean GPU utilization: 85.8369%
- Peak NVML VRAM: 4806.6172 MiB
- End-to-end throughput: 790.3420 images/hour

## Warm-up observations

- 20260723_200342 / generation_total_seconds: warm-up mean 4.9811, steady-state mean 4.2907, difference 16.0913%.
- 20260723_200342 / gpu_energy_j: warm-up mean 149.0300, steady-state mean 150.1458, difference -0.7431%.
