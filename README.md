# UdaciMed - Hardware-Aware Model Optimization

Coursework repository for hardware-aware model optimization: four lessons of
exercises plus the UdaciMed capstone project (chest X-ray pneumonia detection
optimized for production deployment). Every notebook in this repository has
been completed and executed end-to-end; the results below come from those
committed runs.

## Repository layout

```
lesson-1-intro-to-hardware-aware-model-optimization/   profiling & bottleneck diagnosis
lesson-2-designing-model-architectures-for-hardawre-efficiency/   low-rank, attention, mobile CNNs
lesson-3-discover-hardware-acceleration/               TensorRT, LiteRT, ONNX Runtime
lesson-4-combining-efficient-architectures-with-hardware-acceleration/
project/                                               the UdaciMed capstone (3 notebooks)
```

Each notebook was developed on its own branch (`project/01-baseline-analysis`,
`lesson-2/low-rank-factorization`, ...) and merged into `main`.

## Execution environment

Runs were executed on an 8-core x86_64 Linux machine without a GPU
(Python 3.12, torch 2.3.0+cpu, onnxruntime CPU, tensorflow-cpu + ai-edge-litert
for the LiteRT exercise). GPU-specific stages (TensorRT engine builds, CUDA
providers, NVML monitoring) are implemented for the course's reference T4 but
key off hardware availability and print skip notes on this machine. All
timings below are CPU numbers - conservative relative to the T4 target.

## Project results (capstone)

The full pipeline story - baseline profiling, architecture optimization, ONNX
deployment - with the detailed findings lives in
[`project/README.md`](project/README.md#results-and-findings). Headline
numbers from the executed notebooks:

| Stage | Parameters | GFLOPs/sample | Single-image latency | Batch throughput | Sensitivity |
|-------|-----------:|--------------:|---------------------:|-----------------:|------------:|
| Baseline ResNet-18 (eager PyTorch) | 11.18M | 1.82 | 33.10 ms | 38/s | 100.0% (t=0.4) |
| Architecture-optimized (eager PyTorch) | 1.45M | 0.03 | 4.65 ms | 1,224/s | 98.2% (t=0.7) |
| Deployed (ONNX Runtime, FP32) | 1.45M (5.52 MB file) | 0.03 | 0.498 ms | 3,958/s (batch 16) | 98.21% |

**Production SLA scorecard: 5/5 targets met** - memory 5.52 MB (<100 MB),
latency 0.498 ms (<3 ms), throughput 3,958 samples/sec (>2,000), FLOP
reduction 98.5% (>80%), sensitivity 98.21% (>98%). Combined improvement over
the baseline: ~66x latency, ~104x throughput.

The two decisive findings: the baseline silently upscaled 64x64 inputs to
224x224 (12.25x wasted compute - removing it alone cut 91.9% of FLOPs), and
depthwise separable convolutions cut parameters 87% while sensitivity was
recovered above the clinical floor by retraining.

## Lesson exercise results

**Lesson 1 - bottleneck investigation (PyTorch Profiler + NVML).** The
deliberately inefficient churn MLP showed the classic overhead signature:
throughput peaks at a mid-size batch (4.7 ms at batch 64 vs 27 ms at 512 and
68 ms at 1024) and the profiler attributes most time to synchronization
points (`.item()`) and swarms of tiny elementwise kernels rather than the
linear layers themselves. Also fixed a pandas 3 copy-on-write no-op in the
starter preprocessing that left NaNs and crashed training.

**Lesson 2.1 - low-rank factorization.** SVD-initialized factorization of the
two large classifier layers shrank the 10.5 MB edge model to 4.3 MB at rank
ratio 0.25 (2.0 MB at 0.1), comfortably inside the 8 MB deployment budget,
with latency dominated by launch overhead rather than matmul size.

**Lesson 2.2 - transformer attention.** Multi-Query Attention beat standard
Multi-Head Attention by 28% throughput at batch 64 (27.4 vs 21.4 samples/sec)
and 56% at batch 128 on CPU FP32, matching the KV-cache math implemented in
the exercise; FP16 comparisons are gated to Tensor Core hardware because CPU
half-precision emulation measured ~86x slower than FP32.

**Lesson 2.3 - mobile CNN transformation.** Rebuilding the 30.9M-parameter
CNN with a separable stem and inverted residual stages cut it to 811K
parameters (~97%), with the converted convolution stages showing the expected
~88.5% parameter/FLOP reduction.

**Lesson 3.1 - TensorRT.** Transformer baseline benchmarked (best 16.8
samples/sec at batch 16 on CPU), ONNX export verified; the complete FP16
engine-build and pinned-buffer inference path is implemented for the T4 and
skips cleanly here.

**Lesson 3.2 - LiteRT.** All four mobile configurations (fp32 baseline,
XNNPACK, fp16+XNNPACK, multithreaded) were actually converted and benchmarked
with the LiteRT interpreter via tensorflow-cpu + ai-edge-litert.

**Lesson 3.3 - ONNX Runtime cross-platform.** EfficientNet-B0 exported and
benchmarked at 37.6 samples/sec on the default CPU provider; on this shared
8-core box the hand-tuned threading config landed within run noise of the
defaults (ORT already uses all cores), the honest takeaway being that thread
caps matter for multi-tenancy, not raw speed.

**Lesson 4 - combining architecture + hardware.** Early-exit DenseNet with
low-rank heads (7.7% parameter cut) exported to ONNX; ONNX Runtime roughly
doubled eager PyTorch throughput on CPU (11.8 vs 5.8 samples/sec at batch 32),
demonstrating that architecture savings and runtime acceleration compose.
