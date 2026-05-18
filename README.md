# AuraRun

Maps Strava run biometrics: heart rate, pace, elevation, time of day -> to aesthetic descriptors, then renders each run as a generated image.
<img width="1433" height="801" alt="image" src="https://github.com/user-attachments/assets/533f7738-0b24-4d14-b822-d780d8a69a83" />

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/f0c09ca9-d2fd-4f7a-b52f-6477531b6260" width="420"/><br/>
      Test Input Run Data
    </td>
    <td align="center" valign="middle">
      <h1>→</h1>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/9bca2a7b-7202-43bb-a6d6-37f2af6fa15a" width="420"/><br/>
      Generated Artwork
    </td>
  </tr>
</table>

 


## How it works

- **Fetch** the last 30 runs from Strava via OAuth and a bearer token (`fetch_runs.py`), saving the activity records as JSON.
- **Map** each run's biometrics into four aesthetic axes — mood (from heart rate), lighting (from start hour), terrain (from elevation gain), style (from average speed) — using deterministic threshold rules (`mapper.py`).
- **Render** the resulting prompt with Stable Diffusion v1.5 on whichever device is available, picked automatically: CUDA → MPS → CPU (`generate.py`).
- **Log** every generation to a local MLflow experiment: prompt, params, generation time, optional manual 1–5 quality score, and the output image as an artifact.
- **Browse** the results in a Streamlit gallery (`gallery.py`) with a mood filter sidebar, descriptor placards, and per-run scores.

## Tech stack

- **Data ingestion** — Strava REST API, `requests`, `python-dotenv`
- **Descriptor mapping** — Pure Python, no external dependencies
- **Image generation** — Stable Diffusion v1.5 via `diffusers` and PyTorch (MPS / CUDA / CPU)
- **Experiment tracking** — MLflow, local file-based store at `./mlruns/`
- **HTTP API** — FastAPI, Uvicorn, Pydantic
- **Gallery UI** — Streamlit, themed via `.streamlit/config.toml` + injected CSS
- **Testing** — pytest, `TestClient`, `unittest.mock`

## Example prompt logic

The biometric → descriptor mapping lives in `mapper.py`. Five representative rules:

```python
hr > 170                    →  mood     = "intense frenetic energy"
140 <= hr <= 170            →  mood     = "driven purposeful motion"
7 <= hour <= 11             →  lighting = "crisp morning light"
elevation_gain > 200 m      →  terrain  = "dramatic mountain landscape"
average_speed > 3.5 m/s     →  style    = "sharp geometric motion blur"
```

Descriptors are concatenated into the final SD prompt. Missing biometric fields (e.g. heart rate when no monitor was worn) cause the corresponding descriptor to be skipped — the prompt gracefully degrades instead of fabricating a vibe from null data.

## Results

- **18/18 tests passing** — 17 mapper unit tests (including boundary checks at every threshold) and 1 FastAPI endpoint test that mocks the Stable Diffusion call.
- **~40 seconds end-to-end** to generate a 512×512 image on Apple Silicon (MPS, 30 inference steps, 7.5 guidance scale). ~5–10 minutes on CPU; under 10s on consumer NVIDIA GPUs.
- **4 aesthetic axes × 11 mapping rules** — every branch covered by a boundary test at the exact threshold value (HR 140/170, elevation 50/200, speed 3.5, hour 7/17).
- **MLflow tracks every generation** — prompt, descriptors, generation time, optional manual score, and image artifact — enabling structured prompt A/B evaluation across runs.

## Quickstart

```bash
git clone https://github.com/ZainaNadeem/AuraRun.git && cd AuraRun
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then paste your STRAVA_ACCESS_TOKEN
python generate.py --activity_id 1 --score 5
streamlit run gallery.py
```

The bundled `tests/fixtures/sample_runs.json` is used as a fallback so the pipeline works end-to-end before you wire up live Strava data. Use `python fetch_runs.py` once your token is set to pull your own activities.

## Notes

- **Gallery aesthetic** — the Streamlit gallery is themed in a French art gallery style (olive walls, Didot + Garamond serif typography, white-matted framed prints) because the runs-as-art framing felt closer to an exhibition than a dashboard. Pure visual choice; all content text stays in English so the fitness-app context remains readable.
- **Stable Diffusion v1.5 specifically** — small enough to run on Apple Silicon at reasonable speeds, and its default outputs are coherent without negative-prompt tuning. Easy to swap for SDXL or a fine-tune later.
- **Everything runs local** — no remote MLflow server, no cloud GPU, no API keys beyond Strava. Generated images live in `./outputs/`, runs in `./mlruns/`, both gitignored.
