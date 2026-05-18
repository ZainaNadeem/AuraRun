import json
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from generate import generate_artwork
from mapper import build_descriptors, build_prompt

DATA_PATH = Path(__file__).parent / "data" / "sample_runs.json"
OUTPUTS_DIR = Path(__file__).parent / "outputs"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="AuraRun")
app.mount("/images", StaticFiles(directory=OUTPUTS_DIR), name="images")


class GenerateRequest(BaseModel):
    activity_id: str
    score: int | None = None


class GenerateResponse(BaseModel):
    activity_id: str
    image_path: str
    prompt_used: str
    generation_time_seconds: float


class Artwork(BaseModel):
    filename: str
    activity_id: str


def _load_activity_by_id(activity_id: str) -> dict | None:
    if not DATA_PATH.exists():
        return None
    activities = json.loads(DATA_PATH.read_text())
    for activity in activities:
        if str(activity.get("id")) == activity_id:
            return activity
    return None


@app.post("/generate", response_model=GenerateResponse)
def post_generate(req: GenerateRequest) -> GenerateResponse:
    activity = _load_activity_by_id(req.activity_id)
    if activity is None:
        raise HTTPException(
            status_code=404,
            detail=f"activity_id {req.activity_id!r} not found in {DATA_PATH.name}",
        )

    descriptors = build_descriptors(activity)
    prompt = build_prompt(descriptors)

    start = time.perf_counter()
    image_path = generate_artwork(prompt, req.activity_id)
    elapsed = time.perf_counter() - start

    return GenerateResponse(
        activity_id=req.activity_id,
        image_path=image_path,
        prompt_used=prompt,
        generation_time_seconds=elapsed,
    )


@app.get("/artworks", response_model=list[Artwork])
def get_artworks() -> list[Artwork]:
    return [
        Artwork(filename=path.name, activity_id=path.stem)
        for path in sorted(OUTPUTS_DIR.glob("*.png"))
    ]
