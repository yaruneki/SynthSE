from pathlib import Path

from synthse.generation.model_loader import load_text_to_image_pipeline
from synthse.generation.generator import generate_image


def main() -> None:
    model_id = "hf-internal-testing/tiny-stable-diffusion-pipe"
    requested_device = "cuda"

    print("Loading test model...")
    pipeline, actual_device = load_text_to_image_pipeline(
        model_id=model_id,
        device=requested_device,
    )

    print(f"Model loaded on device: {actual_device}")

    output_path = Path("outputs/images/test/smoke_test_seed_42.png")

    result = generate_image(
        pipeline=pipeline,
        prompt="Photo portrait of a Software Engineer that codes",
        seed=42,
        output_path=str(output_path),
        num_inference_steps=2,
        guidance_scale=7.5,
        width=64,
        height=64,
        device=actual_device,
    )

    print("Generation completed.")
    print(result)


if __name__ == "__main__":
    main()