"""Download the public Roboflow datasets used for training (YOLO format)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from autoscout.config import DATA, ROBOFLOW_API_KEY  # noqa: E402

SETS = [
    # (workspace, project, version, purpose)
    ("5419-computer-vision-2026", "new-5419-cv-d6qtg", 4, "robots"),
    ("frc2073", "robot-seg-06wv0", 8, "robots"),
    ("rudransh-bhattacharya", "frc-robot-id", 2, "robots"),
    ("idk-ul6l8", "2026-frc-robots", 3, "robots"),
    ("matrix-autolabeler", "autoscouting-wide-view", 1, "robots"),
    ("chakchuens-workspace", "frc-fuel-zi0wy", 5, "fuel"),
    ("calebs-workspace-kughq", "frc-fuel-2026-8qtjy", 7, "fuel"),
    ("rishis-workspace-ovjuz", "frc-fuel-2026-cvfs6", 1, "fuel"),
    ("blind-assistant-model", "frc2026_fuel", 1, "fuel"),
    # bumper-boxed sets (red/blue box around each visible bumper segment; PLAN.md 11)
    ("frcrobotfinder", "frc-2023-hwue2", 4, "bumpers"),
    ("frcrobotfinder", "ventura-2025", 3, "bumpers"),
    ("frc2073", "seg-rob-2026", 3, "bumpers"),  # bumper polygons -> boxes
    ("benjamins-workspace-h5vao", "frc-bumpers-h0p1g", 4, "bumpers"),
]


def main():
    from roboflow import Roboflow

    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    for ws, proj, ver, purpose in SETS:
        out = DATA / "datasets" / proj
        if (out / "data.yaml").exists():
            print("have", proj)
            continue
        try:
            rf.workspace(ws).project(proj).version(ver).download("yolov8", location=str(out), overwrite=True)
            n = len(list(out.rglob("*.jpg"))) + len(list(out.rglob("*.png")))
            print("ok", proj, n, "images", flush=True)
        except Exception as e:
            print("FAIL", proj, e, flush=True)


if __name__ == "__main__":
    main()
