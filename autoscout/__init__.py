"""auto-scout: automated FRC REBUILT (2026) scouting from match video."""
import os

# OpenBLAS reserves a buffer per thread: ~1.6 GB of commit per process at 24 threads vs ~0.27 GB at 4.
# Must be set before numpy is imported (the project venv's sitecustomize.py does it even earlier).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
