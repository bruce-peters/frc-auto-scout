"""Paths, API keys and game constants."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
VIDEOS = DATA / "videos"
CACHE = DATA / "cache"
MODELS = ROOT / "models"
OUTPUT = ROOT / "output"

load_dotenv(ROOT / ".env")
TBA_API_KEY = os.environ.get("TBA_API_KEY", "")
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")

# ---- REBUILT field (WPILib blue-origin frame, metres) ----
FIELD_LENGTH = 16.541  # 651.22 in
FIELD_WIDTH = 8.069  # 317.677 in
BLUE_HUB = (4.626, 4.0345)  # game manual: 182.11 in from the blue wall
RED_HUB = (11.915, 4.0345)
HUB_OPENING_HEIGHT = 1.829  # m (72 in, front edge of the top opening)
FUEL_DIAMETER = 0.150  # m (5.91 in)

# ---- match timing (match-clock seconds since auto start) ----
AUTO_LEN = 20.0
TELEOP_LEN = 140.0
# teleop periods as (name, start, end) in seconds since teleop start
TELEOP_PERIODS = [
    ("transition", 0.0, 10.0),
    ("shift1", 10.0, 35.0),
    ("shift2", 35.0, 60.0),
    ("shift3", 60.0, 85.0),
    ("shift4", 85.0, 110.0),
    ("endgame", 110.0, 140.0),
]

TOWER_POINTS = {
    "auto": {"None": 0, "Level1": 15, "Level2": 15, "Level3": 15},
    "endgame": {"None": 0, "Level1": 10, "Level2": 20, "Level3": 30},
}

for _d in (DATA, VIDEOS, CACHE, MODELS, OUTPUT):
    _d.mkdir(parents=True, exist_ok=True)
