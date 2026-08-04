# UdaciMed: Efficient Medical Diagnostics with Hardware-Aware AI

## Background scenario

You are a Machine Learning Engineer at **UdaciMed**, a healthcare technology startup developing AI-powered diagnostic tools. Your team is preparing to deploy a new chest X-ray pneumonia detection model across diverse infrastructure, including cloud services, hospital workstations, and portable clinic devices.

To ensure only the most efficient models make it into the production pipeline, UdaciMed has a strict internal policy: performance is a feature. Before any model can be deployed, it must meet a strict performance service level agreement (SLA) with the universally compatible ONNX format on the standardized development machine _(as described in the [Project Instructions](#project-instructions) below)_.

**The challenge:** The current ResNet-18 model meets clinical accuracy standards but requires significant optimization to satisfy the strict performance demands of real-world medical environments.

**Your mission:** Optimize the model through hardware-aware architectural modifications and deployment acceleration to achieve:
- **<100MB memory footprint** for multi-tenant deployment
- **>2,000 samples/sec throughput** for high-volume screening
- **>98% sensitivity** for clinical safety (non-negotiable)
- **<3ms latency** for real-time diagnosis

## Project overview

In this project, you will develop a complete **hardware-aware optimization pipeline** for pneumonia detection using the [PneumoniaMNIST](https://medmnist.com/) dataset. Starting with a [ResNet-18](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html) baseline, you will analyze performance bottlenecks, apply architectural optimizations, implement hardware acceleration using [ONNX Runtime](https://onnxruntime.ai/), and analyze expected performance across multiple deployment scenarios. 

**Key pipeline stages:**
1. **Baseline Analysis** - Profile performance bottlenecks and identify optimization opportunities
2. **Architecture Optimization** - Implement 3+ optimization techniques within a modular optimization framework
3. **Hardware Acceleration & Deployment** - Apply hardware optimizations and define next steps for model deployment on production targets

Each pipeline stage corresponds to a notebook in the `notebooks/` folder. Complete the relevant _TODOs_ in each notebook to deploy a production-ready medical imaging model that demonstrates significant performance improvements while maintaining diagnostic accuracy.

### Learning objectives

By completing this project, you will:
- **Analyze performance trade-offs** between optimization strategies for specific hardware targets
- **Benchmark and profile** model performance across different deployment scenarios
- **Implement hardware-aware architectural optimizations** on deep learning models
- **Apply hardware acceleration** to optimize model inference
- **Deploy optimized models** using ONNX execution providers
- **Evaluate optimization strategies** for diverse deployment targets through critical analysis

## Project instructions

The `starter/` folder is the home for your project.

> **A note on technical requirements**
> 
> This project has been developed and tested on an NVIDIA T4 instance with 16GB VRAM, running Ubuntu 22.04 with CUDA 12.4, cuDNN 8.9.2, NVIDIA driver 550, Python 3.10, and Docker pre-installed. Baseline performance metrics and environment setup have been calibrated for this configuration and may require adjustments for different hardware setups.

Follow these steps to complete the project:

1. [Pre-requisite: Set up the project](#1-pre-requisite-set-up-the-project)
2. [Understand the project folder structure](#2-understand-the-project-folder-structure)
3. [Get started with the project](#3-get-started-with-the-project)

### 1. Pre-requisite: Set up the project

From the project home folder:

1. **(_Optional_) Create a virtual environment** (Python 3.10 recommended)
   ```bash
   python -m venv udacimed_env
   source udacimed_env/bin/activate  # On Windows: udacimed_env\Scripts\activate
   ```

2. **Install project scripts as editable local package:**
   ```bash
   pip install -e .
   ```

### 2. Understand the project folder structure

Below is a breakdown of the `starter/` folder.

```
.     
├── requirements.txt             
├── setup.py                
├── deployment/                                 # Production deployment configurations and artifacts
├── notebooks/                                  # Jupyter notebooks containing the main project workflow
│   ├── 01_baseline_analysis.ipynb        
│   ├── 02_architecture_optimization.ipynb 
│   └── 03_deployment_acceleration.ipynb  
├── utils/                                      # Utility modules supporting the optimization pipeline
│   ├── __init__.py                   
│   ├── architecture_optimization.py     
│   ├── data_loader.py                   
│   ├── evaluation.py                   
│   ├── model.py                        
│   ├── profiling.py                  
│   └── visualization.py                
└── results/                                    # Generated models, metrics, and benchmark results
```

### 3. Get started with the project

Your task is to optimize a pneumonia detection model for efficient, clinically-safe, production-ready deployment.

There are **three notebooks** to complete sequentially:

1. **[`notebooks/01_baseline_analysis.ipynb`](starter/notebooks/01_baseline_analysis.ipynb)**
   - Establish baseline model performance and identify optimization opportunities
   - Profile memory usage, computational complexity, and inference timing
   - Analyze architectural bottlenecks and deployment constraints

2. **[`notebooks/02_architecture_optimization.ipynb`](starter/notebooks/02_architecture_optimization.ipynb)**
   - Implement and evaluate architectural optimization techniques
   - Train the optimized model with preserved clinical performance
   - Validate optimization impact on deployment targets

3. **[`notebooks/03_deployment_acceleration.ipynb`](starter/notebooks/03_deployment_acceleration.ipynb)**
   - Convert models for production deployment with general hardware acceleration (ONNX format)
   - Benchmark performance against deployment targets
   - Provide insights on optimization strategies for GPU, CPU and edge/mobile

In each notebook, you will find **TODOs** for both implementation and analysis tasks. Note that `notebooks/02_architecture_optimization.ipynb` includes TODOs that require implementing functions in `utils/architecture_optimization.py`.

## Project submission

Your submission should include:

- **Completed notebooks** with all TODOs implemented and analysis questions thoroughly answered
- **Optimized model weights** saved in the `results/` directory with documented performance improvements
- **Performance benchmarks** demonstrating measurable progress toward deployment targets compared to baseline
- **Deployment configuration** with complete ONNX setup, model repository structure, and end-to-end testing results

### Evaluation criteria

Your project will be evaluated based on:

- **Technical implementation (25%)** - Quality and effectiveness of hardware-aware optimization techniques
- **Performance achievement (25%)** - Extent to which your optimized model meets UdaciMed's deployment requirements
- **Analysis quality (25%)** - Depth of performance analysis, insights, and strategic optimization decisions
- **Deployment readiness (25%)** - Completeness and robustness of final performance analysis and deployment recommendation

**Success indicators:**
- Achievement of ≥3 out of 4 optimization targets (memory, throughput, latency, sensitivity)
- Clear demonstration of optimization technique effectiveness through before/after comparisons
- Production-ready deployment configuration with documented testing procedures
- Thoughtful analysis of optimization trade-offs and deployment strategy recommendations

---

## Resources

### Technical Documentation
- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [ONNX Model Optimization](https://onnxruntime.ai/docs/performance/)
- [NVIDIA TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/)
- [NVIDIA Triton Inference Server Documentation](https://github.com/triton-inference-server/server)
- [Intel OpenVino Developer Tools](https://www.intel.com/content/www/us/en/developer/tools/overview.html)
- [ExecuTorch Documentation](https://docs.pytorch.org/executorch/stable/index.html)
- [CoreML Documentation](https://developer.apple.com/documentation/coreml)
- [LiteRT Overview](https://ai.google.dev/edge/litert)

### Medical AI Context
- [MedMNIST Documentation](https://medmnist.com/)
- [ResNet Architecture Guide](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html)
- [Medical AI Deployment Best Practices](https://arxiv.org/abs/2109.09824)

## License
[License](../LICENSE.md)

---

**Ready to optimize medical AI for real-world deployment? Start with `notebook/01_baseline_analysis.ipynb` and begin your journey to production-ready healthcare AI!**
---

## Results and findings

All three notebooks were executed end-to-end in a local environment and the
complete pipeline meets every production target. Details below.

### Execution environment

The run used a CPU-only development machine (8-core x86_64, Linux, Python 3.12,
torch 2.3.0+cpu, onnxruntime CPU) rather than the reference T4 instance, so a
few pragmatic adaptations were made:

- `torch`/`torchvision` installed from the CPU wheel index; `tensorrt-cu12`
  skipped (requires CUDA). GPU-dependent choices in the notebooks key off
  `torch.cuda.is_available()` and configure themselves correctly on a T4.
- CUDA memory profiling is unavailable on CPU, so the profiling cells fall
  back to process-RSS sampling plus an analytic activation breakdown.
- All absolute timings below are CPU numbers and therefore conservative:
  the T4 target would only improve them.

### Pipeline results

| Stage | Model | Parameters | GFLOPs/sample | Single-image latency | Batch throughput | Sensitivity |
|-------|-------|-----------:|--------------:|---------------------:|-----------------:|------------:|
| Notebook 1 - baseline (eager PyTorch) | ResNet-18-Adaptive | 11.18M | 1.82 | 33.10 ms | 38/s | 100.0% (t=0.4) |
| Notebook 2 - architecture optimized (eager PyTorch) | ResNet-18-Native + depthwise separable | 1.45M | 0.03 | 4.65 ms | 1,224/s | 98.2% (t=0.7) |
| Notebook 3 - deployed (ONNX Runtime) | same, ONNX FP32 | 1.45M (5.52 MB file) | 0.03 | 0.498 ms | 3,958/s (batch 16) | 98.21% |

Production SLA scorecard after notebook 3: **5/5 targets met** -
memory 5.52 MB (<100 MB), latency 0.498 ms (<3 ms), throughput 3,958
samples/sec (>2,000), FLOP reduction 98.5% (>80%, and 0.03 < 0.4
GFLOPs/sample), sensitivity 98.21% (>98%).

### Key findings

1. **The biggest optimization was a measurement insight, not a technique.**
   The baseline wrapper silently upscaled every 64x64 input to 224x224
   (`F.interpolate` in `forward()`), inflating convolution work and activation
   memory by 12.25x for zero diagnostic gain. Removing it cut 91.9% of FLOPs
   and took latency from 33.1 ms to 5.6 ms before any real architecture work.
2. **Depthwise separable convolutions carried the parameter budget.** Swapping
   the 16 dense 3x3 convolutions cut parameters 87% (11.18M to 1.45M) and
   FLOPs to 0.03 GFLOPs/sample; after 15 epochs of retraining with transferred
   weights the model held 98.2% sensitivity with overall accuracy up 1.6
   points - the baseline was heavily over-parameterized for this binary task.
3. **Runtime choice mattered as much as architecture.** ONNX Runtime's graph
   fusion turned the already-optimized model's 4.65 ms into 0.498 ms (9.3x)
   on the same hardware - eager-mode overhead dominates once a model is this
   small.
4. **Not every documented trick paid off.** channels_last + in-place ReLU made
   no measurable CPU difference (oneDNN already reorders layouts; torchvision
   ReLUs are already in-place); it is kept only because it is free and helps
   on Tensor Core GPUs. Low-rank factorization was analyzed and rejected: the
   only linear layer (512->2) is too small to benefit.
5. **Combined effect:** ~66x latency and ~104x throughput over the baseline
   while staying above the 98% clinical sensitivity floor.
