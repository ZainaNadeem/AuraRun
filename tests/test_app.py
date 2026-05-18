from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import app

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_runs.json"

client = TestClient(app)


def test_post_generate_returns_200(monkeypatch, tmp_path):
    data_file = tmp_path / "sample_runs.json"
    data_file.write_text(FIXTURE_PATH.read_text())
    monkeypatch.setattr("app.DATA_PATH", data_file)

    fake_image = str(tmp_path / "1.png")
    with patch("app.generate_artwork", return_value=fake_image) as mock_generate:
        response = client.post("/generate", json={"activity_id": "1", "score": 5})

    assert response.status_code == 200
    body = response.json()
    assert body["activity_id"] == "1"
    assert body["image_path"] == fake_image
    assert body["prompt_used"]
    assert "generation_time_seconds" in body
    mock_generate.assert_called_once()
