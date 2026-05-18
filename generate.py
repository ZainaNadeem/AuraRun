import argparse
import json
import time
from pathlib import Path

import mlflow
import torch
from diffusers import StableDiffusionPipeline

from mapper import build_descriptors, build_prompt

MODEL_ID = "runwayml/stable-diffusion-v1-5"
OUTPUTS_DIR = Path(__file__).parent / "outputs"
DATA_PATH = Path(__file__).parent / "data" / "sample_runs.json"
FIXTURE_PATH = Path(__file__).parent / "tests" / "fixtures" / "sample_runs.json"

EXPERIMENT_NAME = "aurarun-prompt-eval"
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


def _load_activity(activity_id: str | None) -> dict:
    for source in (DATA_PATH, FIXTURE_PATH):
        if not source.exists():
            continue
        activities = json.loads(source.read_text())
        if not activities:
            continue
        if activity_id is None:
            return activities[0]
        for activity in activities:
            if str(activity.get("id")) == str(activity_id):
                return activity
    if activity_id is None:
        raise RuntimeError("No activities found in data/ or tests/fixtures/.")
    raise RuntimeError(
        f"Activity id {activity_id!r} not found in {DATA_PATH} or {FIXTURE_PATH}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an AuraRun image and log to MLflow.")
    parser.add_argument(
        "--activity_id",
        type=str,
        default=None,
        help="Strava activity id to render. Defaults to the first available activity.",
    )
    parser.add_argument(
        "--score",
        type=int,
        choices=range(1, 6),
        default=None,
        metavar="{1-5}",
        help="Manual quality score for this generation (1-5). Logged as an MLflow metric.",
    )
    args = parser.parse_args()

    activity = _load_activity(args.activity_id)
    activity_id = str(activity.get("id", "unknown"))
    descriptors = build_descriptors(activity)
    prompt = build_prompt(descriptors)

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        mlflow.log_param("prompt_template", prompt)
        mlflow.log_param("activity_id", activity_id)
        mlflow.log_param("num_inference_steps", NUM_INFERENCE_STEPS)
        mlflow.log_param("guidance_scale", GUIDANCE_SCALE)
        mlflow.log_param("mood_descriptor", descriptors.get("mood"))
        mlflow.log_param("lighting_descriptor", descriptors.get("lighting"))
        mlflow.log_param("style_descriptor", descriptors.get("style"))

        print(f"Device:   {detect_device()}")
        print(f"Activity: {activity.get('name')!r} (id={activity_id})")
        print(f"Prompt:   {prompt}")
        print(f"Run ID:   {run.info.run_id}")

        start = time.perf_counter()
        output_path = generate_artwork(prompt, activity_id)
        elapsed = time.perf_counter() - start

        mlflow.log_metric("generation_time_seconds", elapsed)
        if args.score is not None:
            mlflow.log_metric("manual_score", args.score)

        mlflow.log_artifact(output_path)

        print(f"Saved:    {output_path}")
        print(f"Elapsed:  {elapsed:.2f}s")
        if args.score is not None:
            print(f"Score:    {args.score}/5")


if __name__ == "__main__":
    main()
