import torch
from diffusers import StableDiffusionPipeline


def load_text_to_image_pipeline(model_id: str, device: str):
    """
    Load a Stable Diffusion text-to-image pipeline.

    Args:
        model_id: Hugging Face model identifier.
        device: Target device, usually 'cuda' or 'cpu'.

    Returns:
        A loaded Stable Diffusion pipeline.
    """
    torch_dtype = torch.float16 if device == "cuda" and torch.cuda.is_available() else torch.float32
    actual_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"

    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )

    pipeline = pipeline.to(actual_device)
    pipeline.set_progress_bar_config(disable=False)

    return pipeline, actual_device