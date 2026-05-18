import json
from datetime import datetime
from pathlib import Path

import mlflow
import streamlit as st

from mapper import build_descriptors

ROOT = Path(__file__).parent
OUTPUTS_DIR = ROOT / "outputs"
DATA_PATH = ROOT / "data" / "sample_runs.json"
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "sample_runs.json"
EXPERIMENT_NAME = "aurarun-prompt-eval"


@st.cache_data
def load_activities() -> dict:
    for path in (DATA_PATH, FIXTURE_PATH):
        if not path.exists():
            continue
        activities = json.loads(path.read_text())
        if activities:
            return {str(a.get("id")): a for a in activities}
    return {}


@st.cache_data
def load_scores() -> dict:
    try:
        runs = mlflow.search_runs(
            experiment_names=[EXPERIMENT_NAME],
            order_by=["start_time DESC"],
        )
    except Exception:
        return {}
    if runs.empty:
        return {}

    scores: dict[str, float] = {}
    for _, row in runs.iterrows():
        activity_id = row.get("params.activity_id")
        score = row.get("metrics.manual_score")
        if activity_id and score is not None and activity_id not in scores:
            scores[str(activity_id)] = float(score)
    return scores


def format_date(iso_date: str | None) -> str:
    if not iso_date:
        return "Unknown date"
    try:
        return datetime.fromisoformat(iso_date.replace("Z", "+00:00")).strftime("%b %d, %Y")
    except ValueError:
        return iso_date


st.set_page_config(page_title="AuraRun Gallery", layout="wide")
st.title("AuraRun — Your Runs as Art")

activities = load_activities()
scores = load_scores()

mood_filter = st.sidebar.text_input("Filter by mood descriptor").strip().lower()

artworks = []
for img_path in sorted(OUTPUTS_DIR.glob("*.png")):
    activity_id = img_path.stem
    activity = activities.get(activity_id, {})
    descriptors = build_descriptors(activity) if activity else {}
    mood = descriptors.get("mood")

    if mood_filter and (not mood or mood_filter not in mood.lower()):
        continue

    artworks.append(
        {
            "path": img_path,
            "activity_id": activity_id,
            "name": activity.get("name", "Unknown activity"),
            "date": format_date(activity.get("start_date")),
            "mood": mood,
            "lighting": descriptors.get("lighting"),
            "style": descriptors.get("style"),
            "score": scores.get(activity_id),
        }
    )

if not artworks:
    if mood_filter:
        st.info(f"No artworks match mood filter '{mood_filter}'.")
    else:
        st.info("No images in outputs/ yet. Run `python generate.py` first.")
else:
    st.caption(f"Showing {len(artworks)} artwork{'s' if len(artworks) != 1 else ''}")

    cols = st.columns(3)
    for i, art in enumerate(artworks):
        with cols[i % 3]:
            st.image(str(art["path"]), width="stretch")
            st.markdown(f"**{art['name']}**")
            st.caption(art["date"])
            if art["mood"]:
                st.write(f"Mood: {art['mood']}")
            if art["lighting"]:
                st.write(f"Lighting: {art['lighting']}")
            if art["style"]:
                st.write(f"Style: {art['style']}")
            if art["score"] is not None:
                st.write(f"Score: {int(art['score'])}/5")
            st.divider()
