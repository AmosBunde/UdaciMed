"""
Architecture optimization utilities for hardware-aware model optimization in medical imaging.

This module provides comprehensive implementations of modern neural network optimization
techniques specifically designed for clinical deployment scenarios. Focuses on reducing
computational overhead, memory usage, and inference latency while maintaining diagnostic
accuracy for the PneumoniaMNIST binary classification task.

Key optimization strategies:
    - Interpolation Removal: Eliminates computational overhead from resolution upscaling
    - Depthwise Separable Convolutions: Reduces parameters and FLOPs significantly
    - Grouped Convolutions: Parallel channel processing for improved efficiency
    - Inverted Residual Blocks: Mobile-optimized residual architectures
    - Low-Rank Factorization: Matrix decomposition for parameter reduction
    - Channel Optimization: Memory layout and activation optimizations
    - Parameter Sharing: Weight reuse across similar layer configurations
"""

import copy
from typing import Any, Dict, List, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def create_optimized_model(base_model: nn.Module, optimizations: Dict[str, Any]) -> nn.Module:
    """
    Apply selected optimization strategies in order to create a clinically-optimized model.

    Args:
        base_model: Original ResNet model to optimize for clinical deployment
        optimizations: Dictionary specifying which optimizations to apply with parameters:
            - 'interpolation_removal': bool - Remove upscaling overhead (recommended: True)
            - 'depthwise_separable': bool - Apply depthwise separable convolutions
            - 'grouped_conv': bool - Use grouped convolutions for parallel processing
            - 'channel_optimization': bool - Optimize memory layout and activations
            - 'inverted_residuals': bool - Replace blocks with inverted residuals
            - 'lowrank_factorization': bool - Apply matrix factorization to linear layers
            - 'parameter_sharing': bool - Share weights between similar layers
            
    Returns:
        Optimized model with selected techniques applied, ready for clinical deployment
        
    Example:
        >>> base_model = create_baseline_model()
        >>> optimization_config = {
        ...     'interpolation_removal': True,
        ...     'depthwise_separable': True,
        ...     'channel_optimization': True
        ... }
        >>> optimized_model = create_optimized_model(base_model, optimization_config)
        >>> print("Clinical deployment model ready")
    """
    model = copy.deepcopy(base_model)
  
    print("Starting clinical model optimization pipeline...")
    
    # Order matters: resolution comes first because it changes the activation
    # sizes every later decision is based on; block-level rewrites happen before
    # single-layer swaps so new blocks are not converted twice; parameter-level
    # compression follows once the layer inventory is final; memory-layout and
    # in-place tweaks go last since they must see the finished architecture.
    optimization_order = [
        'interpolation_removal',    # architectural: native-resolution processing
        'inverted_residuals',       # block-level rewrite (MobileNetV2-style)
        'depthwise_separable',      # layer-level convolution swap
        'grouped_conv',             # layer-level convolution swap
        'lowrank_factorization',    # parameter-level compression
        'parameter_sharing',        # parameter-level compression
        'channel_optimization',     # hardware: memory layout + in-place activations
    ]
    
    # Optimization function mapping - connects optimization names to their implementation
    # IMPORTANT: Make sure to experiment with different input parameters for each optimization function, if performance is suboptimal
    optimization_functions = {
        'interpolation_removal': lambda m: apply_interpolation_removal_optimization(m),
        'depthwise_separable': lambda m: apply_depthwise_separable_optimization(m),
        'grouped_conv': lambda m: apply_grouped_convolution_optimization(m),
        'channel_optimization': lambda m: apply_channel_optimization(m),
        'inverted_residuals': lambda m: apply_inverted_residual_optimization(m),
        'lowrank_factorization': lambda m: apply_lowrank_factorization(m),
        'parameter_sharing': lambda m: apply_parameter_sharing(m)
    }
    
    # Smart iteration through the defined optimization order
    applied_optimizations = []
    for opt_name in optimization_order:
        # Check if this optimization is requested and available
        if optimizations.get(opt_name, False) and opt_name in optimization_functions:
            print(f"   Applying {opt_name.replace('_', ' ')} optimization...")
            try:
                # Apply the optimization using the mapped function
                model = optimization_functions[opt_name](model)
                applied_optimizations.append(opt_name)
            except Exception as e:
                print(f"   ERROR: {opt_name} optimization failed: {e}")
        elif opt_name not in optimization_functions:
            print(f"   WARNING: Unknown optimization: {opt_name}")
    
    # Report results
    if applied_optimizations:
        print(f"Applied optimizations in order: {' → '.join(applied_optimizations)}")
    else:
        print("No optimizations were applied")
        
    return model

def _replace_module(model: nn.Module, module_name: str, new_module: nn.Module) -> None:
    """Replace a (possibly nested) submodule, addressed by its dotted name."""
    parts = module_name.split('.')
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


class NativeResolutionModel(nn.Module):
    """ResNet wrapper that processes inputs at native resolution.

    The original ResNetBaseline upscales every input to 224x224 before the
    backbone runs, which multiplies convolution work and activation memory by
    (224/64)^2 = 12.25x without adding any information. This wrapper exposes
    the same underlying backbone but feeds it the input as-is.
    """

    def __init__(self, backbone: nn.Module, native_size: int = 64, num_classes: int = 2) -> None:
        super().__init__()
        self.model = backbone  # keep the attribute name so state_dict keys stay compatible
        self.input_size = native_size
        self.target_size = native_size  # no resizing happens anymore
        self.architecture_name = "ResNet-18-Native"
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # No interpolation: the backbone sees the image at its true resolution
        return self.model(x)


class DepthwiseSeparableConv2d(nn.Module):
    """Drop-in replacement for a dense Conv2d: depthwise then pointwise.

    The depthwise stage convolves each input channel with its own k x k kernel
    (groups = in_channels); the pointwise 1x1 stage then mixes information
    across channels. Parameter count falls from in*out*k^2 to in*k^2 + in*out.
    BatchNorm + ReLU sit between the two stages (MobileNet-style) while the
    surrounding block keeps its own norm/activation, so residual connections
    and output shapes are untouched.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = False) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   stride=stride, padding=padding,
                                   groups=in_channels, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.relu(self.bn(self.depthwise(x))))


class InvertedResidualBlock(nn.Module):
    """MobileNetV2-style inverted residual: expand -> depthwise -> project.

    A 1x1 convolution first expands the channels (the "inverted" part - classic
    bottlenecks compress instead), a depthwise 3x3 does the spatial filtering
    cheaply, and a final 1x1 projects back down with no activation (linear
    bottleneck). ReLU6 keeps activations in a quantization-friendly range.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 expand_ratio: int = 6) -> None:
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ]
        layers += [
            # Depthwise 3x3 handles the spatial pattern per channel
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride,
                      padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # Linear projection back to the block's output width
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.block(x)
        return self.block(x)


class LowRankLinear(nn.Module):
    """Linear layer factorized as W ~= U @ V with a rank-r inner dimension.

    Initialized from the truncated SVD of the original weight so the factorized
    layer starts as the best rank-r approximation of what the network already
    learned, instead of from random weights.
    """

    def __init__(self, linear: nn.Linear, rank: int) -> None:
        super().__init__()
        out_features, in_features = linear.weight.shape
        self.first = nn.Linear(in_features, rank, bias=False)
        self.second = nn.Linear(rank, out_features, bias=linear.bias is not None)

        with torch.no_grad():
            u, s, vh = torch.linalg.svd(linear.weight, full_matrices=False)
            sqrt_s = torch.sqrt(s[:rank])
            self.first.weight.copy_(sqrt_s.unsqueeze(1) * vh[:rank])   # (rank, in)
            self.second.weight.copy_(u[:, :rank] * sqrt_s)             # (out, rank)
            if linear.bias is not None:
                self.second.bias.copy_(linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.second(self.first(x))


# --------------------------------------
# INTERPOLATION REMOVAL (NATIVE RESOLUTION)
# --------------------------------------

def apply_interpolation_removal_optimization(model: nn.Module, native_size: int = 64) -> nn.Module:
    """
    Remove interpolation overhead by processing images at native resolution.
    
    Args:
        model: Model with interpolation capability (e.g., ResNetBaseline)
        native_size: Native input resolution to process (64 for clinical deployment)
        
    Returns:
        Optimized model that processes at native resolution without interpolation

    Note: 
        In `data_loader.py`, we would also want to replace ImageNet stats with chest 
        X-ray specific to check if accuracy improves, but you can skip this for simplicity 
        as normalization affects accuracy/sensitivity and not operational efficiency.
        
    Example:
        >>> baseline_model = create_baseline_model()
        >>> optimized_model = apply_interpolation_removal_optimization(baseline_model, 64)
        >>> # Model now processes 64x64 images directly without upscaling
    """
    # Deep copy model to avoid modifying original
    optimized_model = copy.deepcopy(model)

    print(f"Applying native resolution optimization ({native_size}x{native_size})...")
    
    # ResNetBaseline keeps the actual torchvision ResNet under .model; rewrap it
    # so the forward pass skips F.interpolate entirely
    backbone = optimized_model.model if hasattr(optimized_model, 'model') else optimized_model
    num_classes = getattr(optimized_model, 'num_classes', 2)
    optimized_model = NativeResolutionModel(backbone, native_size=native_size, num_classes=num_classes)

    # Report optimization status and provide deployment guidance
    print("INTERPOLATION REMOVAL completed.")
    
    return optimized_model

# --------------------------------------
# DEPTHWISE SEPARABLE CONVOLUTION MODULES
# --------------------------------------

def apply_depthwise_separable_optimization(
    model: nn.Module,
    layer_names: Optional[List[str]] = None,
    min_channels: int = 16,
    preserve_residuals: bool = True
) -> nn.Module:
    """
    Convert suitable Conv2d layers to DepthwiseSeparableConv2d for clinical efficiency.
    
    Systematically replaces standard convolutions with depthwise separable alternatives
    to reduce computational cost and memory usage while preserving diagnostic accuracy.
    Essential for deploying medical imaging models on resource-constrained devices.
    
    Args:
        model: Input model to optimize for clinical deployment
        layer_names: Specific layer names to convert (None = convert all suitable layers)
        min_channels: Minimum input/output channels required for conversion
        preserve_residuals: Use residual-compatible configurations for ResNet models
        
    Returns:
        Optimized model with depthwise separable convolutions applied
        
    Note:
        Only converts layers that benefit from depthwise separation (kernel_size > 1,
        sufficient channels, not already grouped). Preserves ResNet compatibility by
        maintaining residual connection requirements.
        
    Example:
        >>> model = create_baseline_model()
        >>> optimized_model = apply_depthwise_separable_optimization(
        ...     model, min_channels=32
        ... )
        >>> # Suitable Conv2d layers now use depthwise separable convolutions
    """
    # Deep copy model to avoid modifying original
    optimized_model = copy.deepcopy(model)
    replacements = 0  # Track number of successful replacements

    print("Applying depthwise separable convolution optimization...")

    # Collect suitable layers first (mutating while iterating named_modules is unsafe)
    candidates = []
    for name, module in optimized_model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        if layer_names is not None and name not in layer_names:
            continue
        # 1x1 convolutions gain nothing from separation; grouped layers are
        # already factorized; tiny channel counts are not worth the swap
        if (module.kernel_size[0] > 1 and module.groups == 1
                and module.in_channels >= min_channels
                and module.out_channels >= min_channels):
            candidates.append((name, module))

    for name, module in candidates:
        replacement = DepthwiseSeparableConv2d(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size[0],
            stride=module.stride[0],
            padding=module.padding[0],
            bias=module.bias is not None,
        )
        # Same in/out channels and stride, so residual additions still line up
        _replace_module(optimized_model, name, replacement)
        replacements += 1

    # Report optimization status
    if replacements > 0:
        print(f"DEPTHWISE SEPARABLE completed: Successfully applied to layers with {replacements} replacements")
    else:
        print("WARNING: DEPTHWISE SEPARABLE not applied: No suitable layers found for replacement")

    return optimized_model

# --------------------------------------
# GROUPED CONVOLUTION MODULES
# --------------------------------------

def apply_grouped_convolution_optimization(
    model: nn.Module,
    groups: int = 2,
    min_channels: int = 32,
    layer_names: Optional[List[str]] = None,
    do_depthwise: Optional[bool] = False,
) -> nn.Module:
    """
    Convert suitable Conv2d layers to grouped convolutions for parallel efficiency.
    
    Args:
        model: Input model to optimize
        groups: Number of groups for grouped convolution (typically 2-8)
        min_channels: Minimum channels required for conversion
        layer_names: Specific layers to convert (None = all suitable layers)
        do_depthwise: Whether to apply depthwise grouping (groups=in_channels)
        
    Returns:
        Model with grouped convolutions applied for enhanced efficiency
        
    Note:
        Grouped convolutions can be highly efficient on certain hardware backends, 
        especially when used with memory formats like channels_last and mixed precision (AMP)
        
    Example:
        >>> model = create_baseline_model()
        >>> optimized_model = apply_grouped_convolution_optimization(
        ...     model, groups=4, min_channels=64
        ... )
        >>> # Suitable layers now use 4-group parallel processing
    """
    # Deep copy model to avoid modifying original
    optimized_model = copy.deepcopy(model)
    # Track number of successful and skipped replacements
    replacements = 0
    skipped = 0

    print(f"Applying grouped convolution optimization (groups={groups})...")

    candidates = []
    for name, module in optimized_model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        if layer_names is not None and name not in layer_names:
            continue
        layer_groups = module.in_channels if do_depthwise else groups
        # Channels must split evenly across groups, and grouping a 1x1 or
        # already-grouped convolution brings no benefit
        if (module.kernel_size[0] > 1 and module.groups == 1
                and module.in_channels >= min_channels
                and module.in_channels % layer_groups == 0
                and module.out_channels % layer_groups == 0):
            candidates.append((name, module, layer_groups))
        elif module.kernel_size[0] > 1:
            skipped += 1

    for name, module, layer_groups in candidates:
        replacement = nn.Conv2d(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=layer_groups,
            bias=module.bias is not None,
        )
        _replace_module(optimized_model, name, replacement)
        replacements += 1

    # Report optimization status and provide deployment tipes
    if replacements > 0:
        print(f"GROUPED CONV completed: Successfully applied to layers with {replacements} replacements. Skipped {skipped} layers.")
        print("\nDEPLOYMENT TIP: For some hardware (like NVIDIA GPUs), grouped convolutions may require specific memory formats (channels_last) and mixed precision to achieve maximum throughput.")
    else:
        print("WARNING: GROUPED CONV not applied: No suitable layers found for replacement")

    return optimized_model

# --------------------------------------
# INVERTED RESIDUAL BLOCKS
# --------------------------------------

def apply_inverted_residual_optimization(
    model: nn.Module,
    target_layers: Optional[List[str]] = None,
    expand_ratio: int = 6
) -> nn.Module:
    """
    Replace suitable blocks with mobile-optimized InvertedResidual blocks.

    Args:
        model: Original model for mobile optimization
        target_layers: Specific layer names to convert (None = auto-detect suitable blocks)
        expand_ratio: Channel expansion factor for inverted residuals (6 is optimal)
        
    Returns:
        Model with mobile-optimized inverted residual blocks
        
    Note:
        This optimization targets BasicBlock structures and converts them to mobile-friendly
        inverted residuals. Most effective for deployment on edge devices and mobile platforms
        common in point-of-care medical applications.
        
    Example:
        >>> model = create_baseline_model()
        >>> mobile_model = apply_inverted_residual_optimization(
        ...     model, expand_ratio=6
        ... )
        >>> # Suitable blocks now use mobile-optimized inverted residuals
    """
    # Deep copy model to avoid modifying original
    optimized_model = copy.deepcopy(model)
    replacements = 0  # Track number of successful replacements

    print(f"Applying mobile inverted residual optimization...")
    
    from torchvision.models.resnet import BasicBlock

    candidates = []
    for name, module in optimized_model.named_modules():
        if not isinstance(module, BasicBlock):
            continue
        if target_layers is not None and name not in target_layers:
            continue
        candidates.append((name, module))

    for name, module in candidates:
        in_channels = module.conv1.in_channels
        out_channels = module.conv2.out_channels
        stride = module.conv1.stride[0]  # first conv carries the block's stride
        replacement = InvertedResidualBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            expand_ratio=expand_ratio,
        )
        _replace_module(optimized_model, name, replacement)
        replacements += 1

    # Report optimization status
    if replacements > 0:
        print(f"INVERTED RESIDUALS completed: Successfully applied to layers with {replacements} replacements")
    else:
        print("WARNING: INVERTED RESIDUALS not applied: No suitable layers found for replacement")

    return optimized_model

# --------------------------------------
# LOW-RANK FACTORIZATION MODULES
# --------------------------------------

def apply_lowrank_factorization(
    model: nn.Module,
    min_params: int = 10_000,
    rank_ratio: float = 0.25
) -> nn.Module:
    """
    Apply low-rank factorization to large linear layers for parameter reduction.
    
    Args:
        model: Input model to optimize for clinical deployment
        min_params: Minimum parameter count to consider for factorization
        rank_ratio: Fraction of minimum dimension to use as factorization rank
    
    Returns:
        Model with low-rank factorized linear layers for reduced memory usage
        
    Note:
        Only factorizes layers with sufficient parameters to benefit from compression.
        Rank selection balances compression ratio with accuracy preservation - lower
        ranks provide more compression but may impact diagnostic performance.
        
    Example:
        >>> model = create_baseline_model()
        >>> compressed_model = apply_lowrank_factorization(
        ...     model, min_params=5000, rank_ratio=0.5
        ... )
        >>> # Large linear layers now use low-rank factorization
    """
    # Deep copy model to avoid modifying original
    optimized_model = copy.deepcopy(model)
    replacements = 0  # Track number of successful replacements

    print("Applying low-rank factorization optimization...")

    candidates = []
    for name, module in optimized_model.named_modules():
        if isinstance(module, nn.Linear) and module.in_features * module.out_features >= min_params:
            candidates.append((name, module))

    for name, module in candidates:
        rank = max(1, int(min(module.in_features, module.out_features) * rank_ratio))
        # Factorization only pays off while (in + out) * rank < in * out
        if rank * (module.in_features + module.out_features) >= module.in_features * module.out_features:
            continue
        _replace_module(optimized_model, name, LowRankLinear(module, rank))
        replacements += 1

    # Report optimization status
    if replacements > 0:
        print(f"LOW RANK FACTORIZATION completed: Successfully applied to layers with {replacements} replacements")
    else:
        print("WARNING: LOW RANK FACTORIZATION not applied: No suitable layers found for replacement")

    return optimized_model

# --------------------------------------
# CHANNEL OPTIMIZATION FUNCTIONS
# --------------------------------------

def apply_channel_optimization(
    model: nn.Module,
    enable_channels_last: bool = True,
    enable_inplace_relu: bool = True
) -> nn.Module:
    """
    Apply channel-level optimizations for enhanced hardware efficiency.

    Implements memory layout and activation optimizations to improve hardware utilization
    and reduce memory bandwidth requirements.

    Args:
        model: Model to optimize for hardware efficiency
        enable_channels_last: E.g., you'd use NHWC memory layout for faster GPU convolutions
        enable_inplace_relu: Convert ReLU layers to in-place for memory savings
    
    Returns:
        Hardware-optimized model with improved memory efficiency
        
    Note:
        The 'channels last' memory format can significantly improve convolution performance on certain hardware 
        (e.g., modern GPUs with specialized cores) but requires input tensors to be converted...
        
    Example:
        >>> model = create_baseline_model()
        >>> optimized_model = apply_channel_optimization(model)
        >>> # Remember to convert inputs: input.to(memory_format=torch.channels_last)
    """
    # Deep copy model to avoid modifying original
    optimized_model = copy.deepcopy(model)
    
    print("Applying channel-level hardware optimizations...")
    
    if enable_inplace_relu:
        # In-place ReLU overwrites its input buffer instead of allocating a copy
        # of every activation map - free memory savings at inference time
        converted = 0
        for module in optimized_model.modules():
            if isinstance(module, nn.ReLU) and not module.inplace:
                module.inplace = True
                converted += 1
        print(f"   In-place ReLU: {converted} layers converted")

    if enable_channels_last:
        # NHWC layout lets convolution kernels stream memory contiguously along
        # the channel dimension; inputs must be converted the same way, e.g.
        # input.to(memory_format=torch.channels_last)
        optimized_model = optimized_model.to(memory_format=torch.channels_last)
        print("   Memory format: channels_last (remember to convert inputs too)")

    # Report optimization status
    print("CHANNEL OPTIMIZATION completed")

    return optimized_model

# --------------------------------------
# PARAMETER SHARING FUNCTIONS
# --------------------------------------

def apply_parameter_sharing(
    model: nn.Module,
    sharing_groups: Optional[List[List[str]]] = None,
    layer_types: Optional[List[Type[nn.Module]]] = None
) -> nn.Module:
    """
    Apply parameter sharing between layers to reduce memory and improve efficiency.

    Shares weight parameters between layers with identical shapes to reduce memory
    footprint and potentially improve generalization. 

    Args:
        model: Model to optimize through parameter sharing
        sharing_groups: Manual specification of layer groups to share parameters.
                       If None, automatically groups layers with identical weight shapes.
        layer_types: Types of layers to consider for parameter sharing 
                    (defaults to Conv2d for maximum impact)
    
    Returns:
        Memory-optimized model with parameter sharing applied
        
    Note:
        Parameter sharing can improve model generalization by enforcing weight
        consistency across similar layers. Most effective when applied to layers
        with identical computational roles and sufficient parameter count.
        
    Example:
        >>> model = create_baseline_model()
        >>> shared_model = apply_parameter_sharing(model)
        >>> # Layers with identical shapes now share parameters
    """    
    # Default to Conv2d layers (largest parameter count and memory footprint)
    if layer_types is None:
        layer_types = [nn.Conv2d]

    # Deep copy model to avoid modifying original
    optimized_model = copy.deepcopy(model)
    # Track number of sharing layers and shared parameters
    total_shared = 0
    total_parameters_shared = 0
    
    print("Applying parameter sharing optimization...")

    named = dict(optimized_model.named_modules())

    if sharing_groups is None:
        # Auto-group: only layers with identical weight shapes (and conv
        # geometry) can literally reuse the same nn.Parameter instance
        auto_groups: Dict[tuple, List[str]] = {}
        for name, module in optimized_model.named_modules():
            if any(isinstance(module, t) for t in layer_types) and hasattr(module, 'weight'):
                key = (type(module).__name__, tuple(module.weight.shape),
                       getattr(module, 'stride', None), getattr(module, 'padding', None))
                auto_groups.setdefault(key, []).append(name)
        sharing_groups = [names for names in auto_groups.values() if len(names) > 1]

    for group in sharing_groups:
        anchor = named[group[0]]
        for other_name in group[1:]:
            other = named[other_name]
            if other.weight.shape != anchor.weight.shape:
                print(f"   WARNING: skipping {other_name} (shape mismatch with {group[0]})")
                continue
            # Point the duplicate layer at the anchor's parameter object; both
            # layers now read and update the same weights
            other.weight = anchor.weight
            if getattr(other, 'bias', None) is not None and getattr(anchor, 'bias', None) is not None:
                other.bias = anchor.bias
            total_shared += 1
            total_parameters_shared += anchor.weight.numel()
   
    # Report optimization status
    if total_shared > 0:
        print(f"PARAMETER SHARING completed - Successfully shared parameters for {total_shared} layers")
        print(f"   Total parameters shared: {total_parameters_shared:,}")
    else:
        print("WARNING: PARAMETER SHARING failed - No suitable layer groups found for optimization")
    
    return optimized_model