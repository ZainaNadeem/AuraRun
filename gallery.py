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

GALLERY_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,500;1,400&family=Playfair+Display:wght@400;500;600&family=Cormorant+Garamond:ital,wght@0,400;0,500;1,400&display=swap');

/* Olive gallery walls */
.stApp {
    background-color: #5D6648;
}

html, body, [class*="st-"], [data-testid="stMarkdownContainer"] p {
    font-family: 'Garamond', 'EB Garamond', 'Cormorant Garamond', 'Hoefler Text', Georgia, serif;
    color: #F5EFE0;
}

h1, h2, h3, h4 {
    font-family: 'Didot', 'Bodoni 72', 'Playfair Display', 'Cormorant Garamond', Georgia, serif !important;
    font-weight: 500 !important;
    color: #F5EFE0 !important;
    letter-spacing: 0.01em;
}

h1 {
    font-size: 3.5rem !important;
    text-align: center !important;
    margin-top: 2rem !important;
    margin-bottom: 0.25rem !important;
    letter-spacing: 0.04em !important;
}

.gallery-subtitle {
    text-align: center !important;
    font-style: italic;
    color: #C5C2A4;
    font-size: 1.05rem;
    letter-spacing: 0.08em;
    margin-bottom: 0.5rem;
    font-family: 'Garamond', 'EB Garamond', 'Cormorant Garamond', 'Hoefler Text', Georgia, serif;
}

.gallery-rule {
    width: 64px;
    border: none;
    border-top: 1px solid #C5C2A4;
    margin: 1.5rem auto 3rem auto;
}

/* Framed prints — white matte pops against olive wall */
.stImage img {
    border: 1px solid #1F1B17;
    box-shadow: 8px 8px 0 #3F4730;
    background: #FFFFFF;
    padding: 14px;
}

[data-testid="stColumn"] {
    padding: 0 1.25rem;
}

.card-number {
    font-variant: small-caps;
    letter-spacing: 0.2em;
    color: #C5C2A4;
    font-size: 0.75rem;
    margin-top: 1.25rem;
    margin-bottom: 0;
}

.card-title {
    font-family: 'Didot', 'Bodoni 72', 'Playfair Display', 'Cormorant Garamond', Georgia, serif;
    font-size: 1.5rem;
    font-weight: 500;
    color: #F5EFE0;
    margin-top: 0.25rem;
    margin-bottom: 0.1rem;
    line-height: 1.2;
}

.card-date {
    font-style: italic;
    color: #C5C2A4;
    font-size: 0.9rem;
    letter-spacing: 0.04em;
    margin-bottom: 1rem;
}

.placard {
    border-top: 1px solid #C5C2A4;
    padding-top: 0.85rem;
    margin-bottom: 1rem;
    font-size: 0.9rem;
    line-height: 1.7;
    color: #F5EFE0;
}

.placard-label {
    font-variant: small-caps;
    letter-spacing: 0.2em;
    color: #C5C2A4;
    font-size: 0.7rem;
    font-style: normal;
    margin-right: 0.4em;
}

.placard-value {
    font-style: italic;
    color: #F5EFE0;
}

.card-divider {
    border: none;
    border-top: 1px solid #4A5238;
    margin: 1rem 0 2rem 0;
}

/* Deeper olive sidebar */
[data-testid="stSidebar"] {
    background-color: #4A4D33;
    border-right: 1px solid #3F4730;
}

[data-testid="stSidebar"] * {
    color: #F5EFE0;
}

[data-testid="stSidebar"] label {
    font-variant: small-caps;
    letter-spacing: 0.18em;
    font-size: 0.8rem;
    color: #C5C2A4 !important;
}

[data-testid="stSidebar"] input {
    background: #F5EFE0;
    color: #1F1B17 !important;
    border: 1px solid #F5EFE0;
    border-radius: 0;
    font-family: 'Garamond', 'EB Garamond', 'Cormorant Garamond', 'Hoefler Text', Georgia, serif;
    font-style: italic;
}

[data-testid="stCaptionContainer"] {
    text-align: center;
    font-style: italic;
    color: #C5C2A4;
    letter-spacing: 0.05em;
}

[data-testid="stAppHeader"] {
    background: transparent;
}
</style>
"""


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
        return datetime.fromisoformat(iso_date.replace("Z", "+00:00")).strftime("%B %d, %Y")
    except ValueError:
        return iso_date


def render_placard(art: dict) -> str:
    rows = []
    if art["mood"]:
        rows.append(f'<span class="placard-label">Mood</span><span class="placard-value">{art["mood"]}</span>')
    if art["lighting"]:
        rows.append(f'<span class="placard-label">Lighting</span><span class="placard-value">{art["lighting"]}</span>')
    if art["style"]:
        rows.append(f'<span class="placard-label">Style</span><span class="placard-value">{art["style"]}</span>')
    if art["score"] is not None:
        rows.append(f'<span class="placard-label">Score</span><span class="placard-value">{int(art["score"])} / 5</span>')
    if not rows:
        return ""
    return '<div class="placard">' + "<br>".join(rows) + "</div>"


st.set_page_config(page_title="AuraRun — Gallery", layout="wide")
st.markdown(GALLERY_CSS, unsafe_allow_html=True)

activities = load_activities()
scores = load_scores()

st.markdown("<h1>AuraRun</h1>", unsafe_allow_html=True)
st.markdown(
    '<p class="gallery-subtitle">Your Runs as Art</p>',
    unsafe_allow_html=True,
)
st.markdown('<hr class="gallery-rule">', unsafe_allow_html=True)

mood_filter = st.sidebar.text_input("Filter by mood").strip().lower()

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
            "name": activity.get("name", "Untitled"),
            "date": format_date(activity.get("start_date")),
            "mood": mood,
            "lighting": descriptors.get("lighting"),
            "style": descriptors.get("style"),
            "score": scores.get(activity_id),
        }
    )

if not artworks:
    if mood_filter:
        st.info(f'No artworks match "{mood_filter}".')
    else:
        st.info("No artworks in outputs/ yet. Run `python generate.py` first.")
else:
    count_label = f"{len(artworks)} artwork{'s' if len(artworks) != 1 else ''}"
    st.caption(count_label)

    cols = st.columns(3)
    for i, art in enumerate(artworks):
        with cols[i % 3]:
            st.image(str(art["path"]), width="stretch")
            st.markdown(
                f'<p class="card-number">No. {art["activity_id"]}</p>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<p class="card-title">{art["name"]}</p>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<p class="card-date">{art["date"]}</p>',
                unsafe_allow_html=True,
            )
            placard = render_placard(art)
            if placard:
                st.markdown(placard, unsafe_allow_html=True)
            st.markdown('<hr class="card-divider">', unsafe_allow_html=True)
