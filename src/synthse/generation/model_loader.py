from __future__ import annotations

from typing import Any

import torch
from diffusers import DiffusionPipeline


# Opzioni specifiche per repository che forniscono una variante FP16.
#
# Non tutti i modelli pubblicano una cartella/variante chiamata "fp16";
# per questo motivo non la applichiamo indistintamente a ogni modello.
MODEL_LOAD_OPTIONS: dict[str, dict[str, Any]] = {
    "segmind/SSD-1B": {
        "variant": "fp16",
        "use_safetensors": True,
    },
    "segmind/Segmind-Vega": {
        "use_safetensors": True,
    },
    "RunDiffusion/Juggernaut-XL-v9": {
        "variant": "fp16",
        "use_safetensors": True,
    },
}


SUPPORTED_QUANTIZATION_MODES = {
    "none",
    "int8",
}


SUPPORTED_QUANTIZED_COMPONENTS = {
    "unet",
}


def normalize_device(
    requested_device: str,
) -> str:
    """
    Resolve the actual device used for inference.

    CUDA is selected only when it was requested and is available.
    Otherwise, the loader falls back to CPU.
    """
    normalized_device = str(
        requested_device
    ).strip().lower()

    cuda_requested = normalized_device.startswith(
        "cuda"
    )

    if cuda_requested and torch.cuda.is_available():
        return "cuda"

    if cuda_requested and not torch.cuda.is_available():
        print(
            "CUDA was requested but is not available. "
            "Falling back to CPU."
        )

    return "cpu"


def normalize_quantization(
    quantization: str | None,
) -> str:
    """
    Normalize and validate the quantization mode.
    """
    if quantization is None:
        return "none"

    normalized_quantization = str(
        quantization
    ).strip().lower()

    if normalized_quantization in {
        "",
        "null",
        "false",
        "no",
        "off",
    }:
        normalized_quantization = "none"

    if (
        normalized_quantization
        not in SUPPORTED_QUANTIZATION_MODES
    ):
        raise ValueError(
            "Unsupported quantization mode: "
            f"{quantization}. "
            "Supported values: "
            f"{sorted(SUPPORTED_QUANTIZATION_MODES)}"
        )

    return normalized_quantization


def normalize_quantized_components(
    quantized_components: list[str] | str | None,
) -> list[str]:
    """
    Normalize the list of pipeline components to quantize.

    Examples:

        ["unet"] -> ["unet"]
        "unet" -> ["unet"]
        "unet,vae" -> ["unet", "vae"]

    For the Stable Diffusion and SDXL models currently used by
    SynthSE, only UNet quantization is enabled.
    """
    if quantized_components is None:
        return ["unet"]

    if isinstance(
        quantized_components,
        str,
    ):
        components = [
            component.strip().lower()
            for component
            in quantized_components.split(",")
            if component.strip()
        ]

    elif isinstance(
        quantized_components,
        (list, tuple, set),
    ):
        components = [
            str(component).strip().lower()
            for component in quantized_components
            if str(component).strip()
        ]

    else:
        raise TypeError(
            "quantized_components must be a string, "
            "a list of strings, or None."
        )

    if not components:
        return ["unet"]

    # Remove duplicate component names while preserving order.
    unique_components = list(
        dict.fromkeys(components)
    )

    unsupported_components = (
        set(unique_components)
        - SUPPORTED_QUANTIZED_COMPONENTS
    )

    if unsupported_components:
        raise ValueError(
            "Unsupported quantized components: "
            f"{sorted(unsupported_components)}. "
            "The current SynthSE loader supports: "
            f"{sorted(SUPPORTED_QUANTIZED_COMPONENTS)}"
        )

    return unique_components


def get_model_load_options(
    model_id: str,
) -> dict[str, Any]:
    """
    Return repository-specific loading options.

    A copy is returned so that the dictionary can be modified safely
    during fallback attempts.
    """
    return dict(
        MODEL_LOAD_OPTIONS.get(
            model_id,
            {},
        )
    )


def load_pipeline_with_variant_fallback(
    model_id: str,
    load_arguments: dict[str, Any],
) -> Any:
    """
    Load a Diffusers pipeline.

    If a repository-specific FP16 variant is unavailable, retry without
    the `variant` argument. This allows the weights to be loaded and
    converted to the requested torch dtype.
    """
    try:
        return DiffusionPipeline.from_pretrained(
            model_id,
            **load_arguments,
        )

    except OSError as error:
        if "variant" not in load_arguments:
            raise

        failed_variant = load_arguments[
            "variant"
        ]

        print(
            "The requested model variant could not be loaded: "
            f"{failed_variant}"
        )
        print(
            "Retrying without the explicit variant."
        )

        fallback_arguments = dict(
            load_arguments
        )

        fallback_arguments.pop(
            "variant",
            None,
        )

        try:
            return DiffusionPipeline.from_pretrained(
                model_id,
                **fallback_arguments,
            )

        except Exception:
            # Preserve the first error when both loading attempts fail,
            # because it usually contains the most useful repository
            # or variant information.
            raise error


def disable_safety_checker(
    pipeline: Any,
) -> None:
    """
    Disable the safety checker when the loaded pipeline contains one.

    SDXL pipelines may not expose the same safety-checker attributes as
    Stable Diffusion 1.x, so the attributes are checked dynamically.
    """
    if hasattr(
        pipeline,
        "safety_checker",
    ):
        pipeline.safety_checker = None

    if hasattr(
        pipeline,
        "requires_safety_checker",
    ):
        pipeline.requires_safety_checker = False


def enable_memory_optimizations(
    pipeline: Any,
) -> None:
    """
    Enable memory-saving features supported by the loaded pipeline.
    """
    if hasattr(
        pipeline,
        "enable_attention_slicing",
    ):
        pipeline.enable_attention_slicing()

    if hasattr(
        pipeline,
        "enable_vae_slicing",
    ):
        pipeline.enable_vae_slicing()

    if hasattr(
        pipeline,
        "enable_vae_tiling",
    ):
        pipeline.enable_vae_tiling()


def load_standard_pipeline(
    model_id: str,
    actual_device: str,
) -> Any:
    """
    Load a non-quantized text-to-image pipeline.
    """
    torch_dtype = (
        torch.float16
        if actual_device == "cuda"
        else torch.float32
    )

    load_arguments: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        **get_model_load_options(
            model_id
        ),
    }

    print(
        "Loading standard pipeline "
        f"with dtype: {torch_dtype}"
    )

    pipeline = load_pipeline_with_variant_fallback(
        model_id=model_id,
        load_arguments=load_arguments,
    )

    disable_safety_checker(
        pipeline
    )

    enable_memory_optimizations(
        pipeline
    )

    if actual_device == "cuda":
        # CPU offload keeps inactive pipeline components in system RAM
        # and moves them to CUDA only when needed. This is useful for
        # SDXL models on GPUs with limited VRAM.
        if hasattr(
            pipeline,
            "enable_model_cpu_offload",
        ):
            print(
                "Enabling model CPU offload."
            )
            pipeline.enable_model_cpu_offload()

        else:
            print(
                "Model CPU offload is not available. "
                "Moving the full pipeline to CUDA."
            )
            pipeline = pipeline.to(
                actual_device
            )

    else:
        pipeline = pipeline.to(
            actual_device
        )

    return pipeline


def build_int8_quantization_config(
    quantized_components: list[str],
) -> Any:
    """
    Build a TorchAO INT8 weight-only pipeline configuration.

    Imports are local so that standard non-quantized runs do not require
    TorchAO to be initialized by this module.
    """
    try:
        from diffusers import (
            PipelineQuantizationConfig,
            TorchAoConfig,
        )
        from torchao.quantization import (
            Int8WeightOnlyConfig,
        )

    except ImportError as error:
        raise RuntimeError(
            "INT8 quantization was requested, but the required "
            "TorchAO or Diffusers quantization classes are not "
            "available. Install compatible versions with:\n"
            "python -m pip install -U "
            "torchao diffusers transformers accelerate"
        ) from error

    quantization_mapping = {
        component: TorchAoConfig(
            Int8WeightOnlyConfig()
        )
        for component in quantized_components
    }

    return PipelineQuantizationConfig(
        quant_mapping=quantization_mapping
    )


def load_int8_pipeline(
    model_id: str,
    actual_device: str,
    quantized_components: list[str],
) -> Any:
    """
    Load a pipeline using TorchAO INT8 weight-only quantization.

    Quantized pipeline loading is currently enabled only for CUDA.
    """
    if actual_device != "cuda":
        raise RuntimeError(
            "The current SynthSE INT8 loader requires CUDA. "
            "CUDA is unavailable or the configuration requested CPU."
        )

    quantization_config = (
        build_int8_quantization_config(
            quantized_components
        )
    )

    # TorchAO weight-only quantization stores supported weights in INT8
    # while computations use a higher precision dtype.
    compute_dtype = torch.bfloat16

    load_arguments: dict[str, Any] = {
        "torch_dtype": compute_dtype,
        "quantization_config": quantization_config,
        "device_map": "cuda",
        **get_model_load_options(
            model_id
        ),
    }

    print(
        "Loading TorchAO INT8 weight-only pipeline."
    )
    print(
        "Quantized components: "
        f"{quantized_components}"
    )
    print(
        "Compute dtype: "
        f"{compute_dtype}"
    )

    pipeline = load_pipeline_with_variant_fallback(
        model_id=model_id,
        load_arguments=load_arguments,
    )

    disable_safety_checker(
        pipeline
    )

    enable_memory_optimizations(
        pipeline
    )

    # Do not call pipeline.to("cuda") here. The quantized pipeline was
    # already placed through device_map="cuda".
    return pipeline


def load_text_to_image_pipeline(
    model_id: str,
    device: str,
    quantization: str | None = None,
    quantized_components: list[str] | str | None = None,
) -> tuple[Any, str]:
    """
    Load a text-to-image Diffusers pipeline.

    The correct pipeline subclass is selected automatically from the
    model repository configuration.

    Args:
        model_id:
            Hugging Face model identifier.

        device:
            Requested execution device, normally "cuda" or "cpu".

        quantization:
            Quantization mode. Supported values are "none" and "int8".

        quantized_components:
            Pipeline components to quantize. For the current
            Stable Diffusion and SDXL models, use ["unet"].

    Returns:
        A tuple containing:
        - the loaded Diffusers pipeline;
        - the actual execution device.
    """
    actual_device = normalize_device(
        device
    )

    normalized_quantization = (
        normalize_quantization(
            quantization
        )
    )

    normalized_components = (
        normalize_quantized_components(
            quantized_components
        )
    )

    print(
        f"Loading model: {model_id}"
    )
    print(
        f"Requested device: {device}"
    )
    print(
        f"Actual device: {actual_device}"
    )
    print(
        "Quantization: "
        f"{normalized_quantization}"
    )

    if normalized_quantization == "int8":
        pipeline = load_int8_pipeline(
            model_id=model_id,
            actual_device=actual_device,
            quantized_components=(
                normalized_components
            ),
        )

    else:
        pipeline = load_standard_pipeline(
            model_id=model_id,
            actual_device=actual_device,
        )

    pipeline.set_progress_bar_config(
        disable=False
    )

    pipeline_class_name = (
        pipeline.__class__.__name__
    )

    print(
        "Pipeline loaded successfully."
    )
    print(
        "Pipeline class: "
        f"{pipeline_class_name}"
    )

    return pipeline, actual_device