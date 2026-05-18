import json
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline

from mapper import build_descriptors, build_prompt

MODEL_ID = "runwayml/stable-diffusion-v1-5"
OUTPUTS_DIR = Path(__file__).parent / "outputs"
DATA_PATH = Path(__file__).parent / "data" / "sample_runs.json"
FIXTURE_PATH = Path(__file__).parent / "tests" / "fixtures" / "sample_runs.json"

NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 7.5


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


_pipeline: StableDiffusionPipeline | None = None


def get_pipeline() -> StableDiffusionPipeline:
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    device = detect_device()
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipeline = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipeline = pipeline.to(device)
    if device == "mps":
        pipeline.enable_attention_slicing()

    _pipeline = pipeline
    return pipeline


def generate_artwork(prompt: str, activity_id: str) -> str:
    pipeline = get_pipeline()
    result = pipeline(
        prompt,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
    )
    image = result.images[0]

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUTS_DIR / f"{activity_id}.png"
    image.save(output_path)
    return str(output_path)


def _load_first_activity() -> dict:
    if DATA_PATH.exists():
        activities = json.loads(DATA_PATH.read_text())
        if activities:
            return activities[0]
    print(f"No activities in {DATA_PATH}; using fixture {FIXTURE_PATH}")
    activities = json.loads(FIXTURE_PATH.read_text())
    return activities[0]


def main() -> None:
    activity = _load_first_activity()
    activity_id = str(activity.get("id", "unknown"))
    descriptors = build_descriptors(activity)
    prompt = build_prompt(descriptors)

    print(f"Device:   {detect_device()}")
    print(f"Activity: {activity.get('name')!r} (id={activity_id})")
    print(f"Prompt:   {prompt}")

    output_path = generate_artwork(prompt, activity_id)
    print(f"Saved:    {output_path}")


if __name__ == "__main__":
    main()
