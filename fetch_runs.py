import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

STRAVA_API_BASE = "https://www.strava.com/api/v3"
ACTIVITIES_ENDPOINT = f"{STRAVA_API_BASE}/athlete/activities"

FIELDS = [
    "id",
    "name",
    "start_date",
    "elapsed_time",
    "distance",
    "average_heartrate",
    "max_heartrate",
    "total_elevation_gain",
    "average_speed",
]


def fetch_recent_runs(access_token: str, limit: int = 30) -> list[dict]:
    headers = {"Authorization": f"Bearer {access_token}"}
    runs: list[dict] = []
    page = 1

    while len(runs) < limit:
        response = requests.get(
            ACTIVITIES_ENDPOINT,
            headers=headers,
            params={"per_page": 100, "page": page},
            timeout=30,
        )
        response.raise_for_status()
        activities = response.json()

        if not activities:
            break

        for activity in activities:
            if activity.get("type") != "Run":
                continue
            runs.append({field: activity.get(field) for field in FIELDS})
            if len(runs) >= limit:
                break

        page += 1

    return runs


def main() -> None:
    load_dotenv()
    access_token = os.getenv("STRAVA_ACCESS_TOKEN")
    if not access_token:
        raise RuntimeError("STRAVA_ACCESS_TOKEN missing from environment")

    runs = fetch_recent_runs(access_token)

    output_path = Path(__file__).parent / "data" / "sample_runs.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(runs, indent=2))
    print(f"Saved {len(runs)} runs to {output_path}")


if __name__ == "__main__":
    main()
