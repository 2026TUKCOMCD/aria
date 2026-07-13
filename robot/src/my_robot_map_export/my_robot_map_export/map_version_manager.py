import json
import os
import re
from datetime import datetime, timezone, timedelta


def find_next_version(output_dir: str, map_id: str) -> int:
    pattern = re.compile(rf"^{re.escape(map_id)}_v(\d+)\.json$")
    versions = []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        return 1

    for filename in os.listdir(output_dir):
        match = pattern.match(filename)
        if match:
            versions.append(int(match.group(1)))

    if not versions:
        return 1

    return max(versions) + 1


def update_map_index(output_dir: str, map_id: str, version: int, image_filename: str, json_filename: str):
    index_path = os.path.join(output_dir, "map_index.json")

    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)
    else:
        index_data = {"maps": []}

    kst = timezone(timedelta(hours=9))
    created_at = datetime.now(kst).isoformat()

    index_data["maps"].append({
        "map_id": map_id,
        "version": version,
        "created_at": created_at,
        "image": image_filename,
        "metadata": json_filename
    })

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    output_dir = "/srv/aria/users/hs/aria/robot/maps_export"
    map_id = "aria_map_clean"

    next_version = find_next_version(output_dir, map_id)
    image_filename = f"{map_id}_v{next_version}.png"
    json_filename = f"{map_id}_v{next_version}.json"

    update_map_index(output_dir, map_id, next_version, image_filename, json_filename)

    print(f"next_version = {next_version}")
    print(f"updated map_index.json")
