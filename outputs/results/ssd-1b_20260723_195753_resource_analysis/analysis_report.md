# SynthSE resource metrics analysis

## Experiment 20260723_195753

- Model: `segmind/SSD-1B`
- Quantization: `none`
- Device: `cuda`
- GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`
- Successful images: 36
- Failed images: 0
- Mean generation time: 6.2572 s
- Median generation time: 6.0377 s
- Mean GPU energy per generated image: 170.6247 J
- Total generation GPU energy: 6142.4890 J
- Total model-load GPU energy: 53.1590 J
- End-to-end GPU energy: 6195.6480 J
- End-to-end GPU energy per image: 172.1013 J
- Mean GPU power: 25.7832 W
- Mean GPU utilization: 71.9149%
- Peak NVML VRAM: 3220.6172 MiB
- End-to-end throughput: 562.9268 images/hour

## Warm-up observations

- 20260723_195753 / generation_total_seconds: warm-up mean 11.9738, steady-state mean 6.0939, difference 96.4889%.
- 20260723_195753 / gpu_energy_j: warm-up mean 226.0880, steady-state mean 169.0400, difference 33.7482%.
