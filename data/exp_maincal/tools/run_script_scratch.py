"""Run scripts/recalibrate_main.py with a scratch cache (data/exp_maincal/cache_script), so data/cache is untouched.
Usage: run_script_scratch.py <recalibrate_main args...>"""
import runpy
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\bruce\Documents\Coding\auto-scout")
sys.path.insert(0, str(ROOT))
from autoscout import config  # noqa: E402

config.CACHE = ROOT / "data" / "exp_maincal" / "cache_script"
sys.argv = [str(ROOT / "scripts" / "recalibrate_main.py")] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
