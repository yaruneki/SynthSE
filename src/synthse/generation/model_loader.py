import torch
from diffusers import StableDiffusionPipeline


def load_text_to_image_pipeline(model_id: str, device: str):
    """
    Load a Stable Diffusion text-to-image pipeline.

    Args:
        model_id: Hugging Face model identifier.
        device: Requested device, usually 'cuda' or 'cpu'.

    Returns:
        A tuple containing the loaded pipeline and the actual device used.
    """
    cuda_available = torch.cuda.is_available()
    actual_device = "cuda" if device == "cuda" and cuda_available else "cpu"
    torch_dtype = torch.float16 if actual_device == "cuda" else torch.float32

    print(f"Loading model: {model_id}")
    print(f"Requested device: {device}")
    print(f"Actual device: {actual_device}")

    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )

    pipeline = pipeline.to(actual_device)
    pipeline.set_progress_bar_config(disable=False)

    if actual_device == "cuda":
        pipeline.enable_attention_slicing()

    return pipeline, actual_device