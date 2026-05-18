from datetime import datetime


def build_descriptors(activity: dict) -> dict:
    descriptors: dict = {}

    hr = activity.get("average_heartrate")
    if hr is not None:
        if hr > 170:
            descriptors["mood"] = "intense frenetic energy"
        elif hr >= 140:
            descriptors["mood"] = "driven purposeful motion"
        else:
            descriptors["mood"] = "meditative flowing calm"

    start_date = activity.get("start_date")
    if start_date:
        hour = datetime.fromisoformat(start_date.replace("Z", "+00:00")).hour
        if hour < 7:
            descriptors["lighting"] = "golden hour dawn mist"
        elif hour <= 11:
            descriptors["lighting"] = "crisp morning light"
        elif hour < 17:
            descriptors["lighting"] = "bright midday clarity"
        else:
            descriptors["lighting"] = "warm dusk atmosphere"

    elevation = activity.get("total_elevation_gain")
    if elevation is not None:
        if elevation > 200:
            descriptors["terrain"] = "dramatic mountain landscape"
        elif elevation >= 50:
            descriptors["terrain"] = "rolling hills texture"
        else:
            descriptors["terrain"] = "flat urban geometry"

    speed = activity.get("average_speed")
    if speed is not None:
        if speed > 3.5:
            descriptors["style"] = "sharp geometric motion blur"
        else:
            descriptors["style"] = "soft impressionist brushstroke"

    return descriptors


def build_prompt(descriptors: dict) -> str:
    parts = [
        descriptors.get("mood"),
        descriptors.get("lighting"),
        descriptors.get("terrain"),
        descriptors.get("style"),
    ]
    return ", ".join(part for part in parts if part)
