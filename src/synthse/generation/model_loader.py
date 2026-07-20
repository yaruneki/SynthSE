from __future__ import annotations

from typing import Any

import torch
from diffusers import (
    DiffusionPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
)


# Modelli basati su Stable Diffusion XL.
#
# Le opzioni indicate qui dipendono dai file presenti nei rispettivi
# repository Hugging Face. In particolare, "variant": "fp16" richiede
# checkpoint con nomi come:
#
#   diffusion_pytorch_model.fp16.safetensors
#
SDXL_MODEL_OPTIONS: dict[str, dict[str, Any]] = {
    "segmind/SSD-1B": {
        "use_safetensors": True,
        "variant": "fp16",
    },
    "segmind/Segmind-Vega": {
        "use_safetensors": True,
    },
    "RunDiffusion/Juggernaut-XL-v9": {
        "use_safetensors": True,
        "variant": "fp16",
    },
}


# Modelli basati sulla pipeline Stable Diffusion classica.
SD_MODEL_OPTIONS: dict[str, dict[str, Any]] = {
    "runwayml/stable-diffusion-v1-5": {
        "use_safetensors": True,
    },
    "stabilityai/stable-diffusion-2-1": {
        "use_safetensors": True,
    },
    "stabilityai/stable-diffusion-2-1-base": {
        "use_safetensors": True,
    },
}


SUPPORTED_QUANTIZATION_MODES = {
    "none",
    "int8",
}


SUPPORTED_QUANTIZED_COMPONENTS = {
    "unet",
    "transformer",
    "text_encoder",
    "text_encoder_2",
}


def resolve_device(requested_device: str) -> str:
    """
    Resolve the requested execution device.

    If CUDA is requested but unavailable, execution falls back to CPU.
    """
    normalized_device = requested_device.strip().lower()

    if normalized_device == "cuda":
        if torch.cuda.is_available():
            return "cuda"

        print(
            "CUDA richiesta ma non disponibile. "
            "Verrà utilizzata la CPU."
        )
        return "cpu"

    if normalized_device == "cpu":
        return "cpu"

    raise ValueError(
        f"Dispositivo non supportato: {requested_device}. "
        "I valori ammessi sono 'cuda' e 'cpu'."
    )


def get_torch_dtype(actual_device: str) -> torch.dtype:
    """
    Return the default dtype for a non-quantized pipeline.
    """
    if actual_device == "cuda":
        return torch.float16

    return torch.float32


def normalize_quantization(
    quantization: str | None,
) -> str:
    """
    Normalize and validate the requested quantization mode.
    """
    normalized_quantization = (
        str(quantization).strip().lower()
        if quantization is not None
        else "none"
    )

    if not normalized_quantization:
        normalized_quantization = "none"

    if normalized_quantization not in SUPPORTED_QUANTIZATION_MODES:
        supported_values = ", ".join(
            sorted(SUPPORTED_QUANTIZATION_MODES)
        )

        raise ValueError(
            f"Quantizzazione non supportata: "
            f"{normalized_quantization}. "
            f"Valori ammessi: {supported_values}."
        )

    return normalized_quantization


def normalize_quantized_components(
    quantized_components: list[str] | str | None,
) -> list[str]:
    """
    Normalize and validate pipeline components to quantize.

    If no components are specified, the UNet is quantized by default.
    This is appropriate for Stable Diffusion and SDXL pipelines.
    """
    if quantized_components is None:
        components = ["unet"]

    elif isinstance(quantized_components, str):
        components = [
            component.strip().lower()
            for component in quantized_components.split(",")
            if component.strip()
        ]

    else:
        components = [
            str(component).strip().lower()
            for component in quantized_components
            if str(component).strip()
        ]

    if not components:
        raise ValueError(
            "È stata richiesta la quantizzazione, ma non è stato "
            "specificato alcun componente da quantizzare."
        )

    unsupported_components = (
        set(components) - SUPPORTED_QUANTIZED_COMPONENTS
    )

    if unsupported_components:
        supported_values = ", ".join(
            sorted(SUPPORTED_QUANTIZED_COMPONENTS)
        )

        raise ValueError(
            "Componenti non supportati per la quantizzazione: "
            f"{sorted(unsupported_components)}. "
            f"Valori ammessi: {supported_values}."
        )

    # Remove duplicates while preserving the original order.
    return list(dict.fromkeys(components))


def build_model_options(
    configured_options: dict[str, Any],
    actual_device: str,
) -> dict[str, Any]:
    """
    Build model loading options according to the execution device.

    An FP16 checkpoint variant should only be explicitly requested
    when CUDA is used.
    """
    model_options = configured_options.copy()

    if actual_device != "cuda":
        model_options.pop("variant", None)

    return model_options


def get_model_options(
    model_id: str,
    actual_device: str,
) -> dict[str, Any]:
    """
    Return repository-specific loading options for a known model.

    Unknown models receive an empty options dictionary and are loaded
    through DiffusionPipeline auto-detection.
    """
    if model_id in SDXL_MODEL_OPTIONS:
        return build_model_options(
            configured_options=SDXL_MODEL_OPTIONS[model_id],
            actual_device=actual_device,
        )

    if model_id in SD_MODEL_OPTIONS:
        return build_model_options(
            configured_options=SD_MODEL_OPTIONS[model_id],
            actual_device=actual_device,
        )

    return {}


def load_sdxl_pipeline(
    model_id: str,
    actual_device: str,
    torch_dtype: torch.dtype,
) -> StableDiffusionXLPipeline:
    """
    Load an SDXL-compatible pipeline without quantization.
    """
    model_options = get_model_options(
        model_id=model_id,
        actual_device=actual_device,
    )

    return StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        token=False,
        low_cpu_mem_usage=True,
        **model_options,
    )


def load_stable_diffusion_pipeline(
    model_id: str,
    actual_device: str,
    torch_dtype: torch.dtype,
) -> StableDiffusionPipeline:
    """
    Load a Stable Diffusion 1.x or 2.x pipeline without quantization.
    """
    model_options = get_model_options(
        model_id=model_id,
        actual_device=actual_device,
    )

    return StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        token=False,
        low_cpu_mem_usage=True,
        **model_options,
    )


def load_generic_pipeline(
    model_id: str,
    torch_dtype: torch.dtype,
) -> DiffusionPipeline:
    """
    Load an unregistered model through pipeline auto-detection.

    This works when the model repository contains a valid
    Diffusers model_index.json.
    """
    print(
        f"Modello non presente nelle configurazioni esplicite: "
        f"{model_id}."
    )
    print(
        "Tentativo di caricamento automatico con "
        "DiffusionPipeline."
    )

    return DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        token=False,
        low_cpu_mem_usage=True,
    )


def build_torchao_quantization_config(
    quantized_components: list[str],
) -> Any:
    """
    Build a TorchAO INT8 weight-only configuration.

    Imports are intentionally local so that TorchAO is loaded only
    when quantization is actually requested. Non-quantized runs do
    not require torchao to be installed.
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
            "La quantizzazione INT8 richiede TorchAO e una "
            "versione compatibile di Diffusers. Installa o aggiorna "
            "i pacchetti con:\n\n"
            "python -m pip install -U torchao diffusers accelerate"
        ) from error

    quant_mapping: dict[str, Any] = {}

    for component in quantized_components:
        quant_mapping[component] = TorchAoConfig(
            Int8WeightOnlyConfig()
        )

    return PipelineQuantizationConfig(
        quant_mapping=quant_mapping,
    )


def load_quantized_pipeline(
    model_id: str,
    actual_device: str,
    quantization: str,
    quantized_components: list[str] | str | None,
) -> DiffusionPipeline:
    """
    Load a pipeline with TorchAO INT8 weight-only quantization.

    The selected pipeline components are quantized during loading.
    Components not selected remain in bfloat16.

    CPU offload is deliberately not applied after loading because the
    quantized pipeline is already placed through device_map.
    """
    if quantization != "int8":
        raise ValueError(
            f"Modalità di quantizzazione non gestita: "
            f"{quantization}"
        )

    if actual_device != "cuda":
        raise RuntimeError(
            "La configurazione TorchAO INT8 attuale richiede CUDA."
        )

    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "La GPU o la build PyTorch corrente non supporta "
            "bfloat16, necessario per questa configurazione "
            "TorchAO INT8."
        )

    normalized_components = normalize_quantized_components(
        quantized_components
    )

    quantization_config = (
        build_torchao_quantization_config(
            quantized_components=normalized_components,
        )
    )

    model_options = get_model_options(
        model_id=model_id,
        actual_device=actual_device,
    )

    print("Quantization backend: TorchAO")
    print("Quantization mode: INT8 weight-only")
    print(
        "Quantized components: "
        f"{', '.join(normalized_components)}"
    )
    print("Compute dtype: torch.bfloat16")

    pipeline = DiffusionPipeline.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        token=False,
        device_map="cuda",
        low_cpu_mem_usage=True,
        **model_options,
    )

    configure_quantized_pipeline(pipeline)

    return pipeline


def configure_vae_memory_optimizations(
    pipeline: Any,
) -> None:
    """
    Enable VAE slicing and tiling when supported.
    """
    vae = getattr(pipeline, "vae", None)

    if vae is None:
        return

    if hasattr(vae, "enable_slicing"):
        vae.enable_slicing()
        print("VAE slicing enabled.")

    if hasattr(vae, "enable_tiling"):
        vae.enable_tiling()
        print("VAE tiling enabled.")


def configure_attention_slicing(
    pipeline: Any,
) -> None:
    """
    Enable attention slicing when supported.
    """
    if hasattr(pipeline, "enable_attention_slicing"):
        pipeline.enable_attention_slicing()
        print("Attention slicing enabled.")


def configure_quantized_pipeline(
    pipeline: Any,
) -> None:
    """
    Configure memory optimizations compatible with a quantized pipeline.

    Model CPU offload is not enabled here because the quantized pipeline
    has already been distributed with device_map='cuda'.
    """
    configure_vae_memory_optimizations(pipeline)
    configure_attention_slicing(pipeline)

    if hasattr(pipeline, "set_progress_bar_config"):
        pipeline.set_progress_bar_config(
            disable=True
        )


def configure_cuda_pipeline(
    pipeline: Any,
) -> None:
    """
    Configure a non-quantized pipeline for a CUDA GPU with limited VRAM.

    Model CPU offload moves inactive components to system memory and
    allows larger SDXL models to run on GPUs with limited VRAM.
    """
    configure_vae_memory_optimizations(pipeline)
    configure_attention_slicing(pipeline)

    try:
        pipeline.enable_xformers_memory_efficient_attention()
        print(
            "xFormers memory efficient attention enabled."
        )

    except Exception:
        # xFormers is optional.
        pass

    try:
        pipeline.enable_model_cpu_offload()
        print("Model CPU offload enabled.")

    except Exception as error:
        print(
            "Model CPU offload non disponibile. "
            "La pipeline verrà caricata completamente sulla GPU."
        )
        print(f"Dettaglio: {error}")

        pipeline.to("cuda")


def load_text_to_image_pipeline(
    model_id: str,
    device: str,
    quantization: str | None = None,
    quantized_components: list[str] | str | None = None,
) -> tuple[Any, str]:
    """
    Load and configure a text-to-image pipeline.

    Supported non-quantized model families:
    - Stable Diffusion 1.x
    - Stable Diffusion 2.x
    - Stable Diffusion XL
    - SSD-1B
    - Segmind Vega
    - JuggernautXL
    - generic Diffusers-compatible repositories

    Supported quantization:
    - none
    - int8, through TorchAO weight-only quantization

    Quantizable pipeline components:
    - unet
    - transformer
    - text_encoder
    - text_encoder_2
    """
    actual_device = resolve_device(device)

    normalized_quantization = normalize_quantization(
        quantization
    )

    print(f"Loading text-to-image model: {model_id}")
    print(f"Resolved device: {actual_device}")
    print(
        f"Quantization requested: "
        f"{normalized_quantization}"
    )

    if normalized_quantization == "int8":
        pipeline = load_quantized_pipeline(
            model_id=model_id,
            actual_device=actual_device,
            quantization=normalized_quantization,
            quantized_components=quantized_components,
        )

        return pipeline, actual_device

    torch_dtype = get_torch_dtype(actual_device)

    print(f"Torch dtype: {torch_dtype}")

    if model_id in SDXL_MODEL_OPTIONS:
        pipeline = load_sdxl_pipeline(
            model_id=model_id,
            actual_device=actual_device,
            torch_dtype=torch_dtype,
        )

    elif model_id in SD_MODEL_OPTIONS:
        pipeline = load_stable_diffusion_pipeline(
            model_id=model_id,
            actual_device=actual_device,
            torch_dtype=torch_dtype,
        )

    else:
        pipeline = load_generic_pipeline(
            model_id=model_id,
            torch_dtype=torch_dtype,
        )

    if actual_device == "cuda":
        configure_cuda_pipeline(pipeline)

    else:
        pipeline.to("cpu")

    if hasattr(pipeline, "set_progress_bar_config"):
        pipeline.set_progress_bar_config(
            disable=True
        )

    return pipeline, actual_device