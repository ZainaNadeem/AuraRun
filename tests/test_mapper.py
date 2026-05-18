import json
from pathlib import Path

import pytest

from mapper import build_descriptors, build_prompt

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_runs.json"


@pytest.fixture(scope="module")
def runs() -> list[dict]:
    return json.loads(FIXTURE_PATH.read_text())


def test_high_hr_morning_flat_fast(runs):
    d = build_descriptors(runs[0])
    assert d["mood"] == "intense frenetic energy"
    assert d["lighting"] == "crisp morning light"
    assert d["terrain"] == "rolling hills texture"
    assert d["style"] == "sharp geometric motion blur"


def test_low_hr_midday_mountain_slow(runs):
    d = build_descriptors(runs[1])
    assert d["mood"] == "meditative flowing calm"
    assert d["lighting"] == "bright midday clarity"
    assert d["terrain"] == "dramatic mountain landscape"
    assert d["style"] == "soft impressionist brushstroke"


def test_mid_hr_dusk_flat_slow(runs):
    d = build_descriptors(runs[2])
    assert d["mood"] == "driven purposeful motion"
    assert d["lighting"] == "warm dusk atmosphere"
    assert d["terrain"] == "flat urban geometry"
    assert d["style"] == "soft impressionist brushstroke"


def test_missing_hr_dawn_hills(runs):
    d = build_descriptors(runs[3])
    assert "mood" not in d
    assert d["lighting"] == "golden hour dawn mist"
    assert d["terrain"] == "rolling hills texture"
    assert d["style"] == "soft impressionist brushstroke"


def test_sparse_activity_yields_no_descriptors(runs):
    assert build_descriptors(runs[4]) == {}


def test_hr_boundary_170_is_driven():
    assert build_descriptors({"average_heartrate": 170.0})["mood"] == "driven purposeful motion"


def test_hr_boundary_140_is_driven():
    assert build_descriptors({"average_heartrate": 140.0})["mood"] == "driven purposeful motion"


def test_hr_boundary_139_is_calm():
    assert build_descriptors({"average_heartrate": 139.0})["mood"] == "meditative flowing calm"


def test_elevation_boundary_200_is_hills():
    assert build_descriptors({"total_elevation_gain": 200.0})["terrain"] == "rolling hills texture"


def test_elevation_boundary_49_is_flat():
    assert build_descriptors({"total_elevation_gain": 49.0})["terrain"] == "flat urban geometry"


def test_speed_boundary_3_5_is_soft():
    assert build_descriptors({"average_speed": 3.5})["style"] == "soft impressionist brushstroke"


def test_lighting_boundary_hour_7_is_morning():
    assert build_descriptors({"start_date": "2024-06-15T07:00:00Z"})["lighting"] == "crisp morning light"


def test_lighting_boundary_hour_17_is_dusk():
    assert build_descriptors({"start_date": "2024-06-15T17:00:00Z"})["lighting"] == "warm dusk atmosphere"


def test_prompt_concatenates_all_descriptors():
    descriptors = {
        "mood": "intense frenetic energy",
        "lighting": "crisp morning light",
        "terrain": "rolling hills texture",
        "style": "sharp geometric motion blur",
    }
    assert build_prompt(descriptors) == (
        "intense frenetic energy, crisp morning light, "
        "rolling hills texture, sharp geometric motion blur"
    )


def test_prompt_skips_missing_keys():
    descriptors = {"mood": "meditative flowing calm", "terrain": "flat urban geometry"}
    assert build_prompt(descriptors) == "meditative flowing calm, flat urban geometry"


def test_prompt_empty_descriptors_returns_empty_string():
    assert build_prompt({}) == ""


def test_full_pipeline_over_fixture(runs):
    for run in runs:
        prompt = build_prompt(build_descriptors(run))
        assert isinstance(prompt, str)
