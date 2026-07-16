from __future__ import annotations

from PIL import Image
import numpy as np


def _to_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32)


def _to_image(array: np.ndarray) -> Image.Image:
    array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def apply_gaussian_noise(
    image: Image.Image,
    std: float,
    rng: np.random.Generator,
) -> Image.Image:
    """
    Apply additive Gaussian noise to an RGB image.

    Args:
        image: Input PIL image.
        std: Standard deviation of the Gaussian noise in pixel intensity units.
        rng: Numpy random generator.

    Returns:
        Corrupted PIL image.
    """
    array = _to_array(image)
    noise = rng.normal(loc=0.0, scale=std, size=array.shape)
    corrupted = array + noise
    return _to_image(corrupted)


def apply_salt_pepper_noise(
    image: Image.Image,
    amount: float,
    rng: np.random.Generator,
) -> Image.Image:
    """
    Apply salt-and-pepper noise to an RGB image.

    Args:
        image: Input PIL image.
        amount: Fraction of pixels to corrupt.
        rng: Numpy random generator.

    Returns:
        Corrupted PIL image.
    """
    array = _to_array(image)
    height, width, _ = array.shape

    mask = rng.random((height, width))
    salt_mask = mask < (amount / 2.0)
    pepper_mask = (mask >= (amount / 2.0)) & (mask < amount)

    array[salt_mask] = 255
    array[pepper_mask] = 0

    return _to_image(array)


def apply_noise(
    image: Image.Image,
    noise_type: str,
    parameters: dict,
    rng: np.random.Generator,
) -> Image.Image:
    """
    Dispatch function for image corruption.
    """
    if noise_type == "gaussian":
        return apply_gaussian_noise(
            image=image,
            std=float(parameters["std"]),
            rng=rng,
        )

    if noise_type == "salt_pepper":
        return apply_salt_pepper_noise(
            image=image,
            amount=float(parameters["amount"]),
            rng=rng,
        )

    raise ValueError(f"Unsupported noise type: {noise_type}")