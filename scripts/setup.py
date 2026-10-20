"""One-command setup: Python environment, data packs and match videos.

  python scripts/setup.py               # .venv + packages + starter data (~0.3 GB download)
  python scripts/setup.py --full        # + cached scans, 60 Hz fuel logs and all 131 match videos (~30 GB)
  python scripts/setup.py --videos 5    # starter + the first 5 match videos (or: all, or 2026cc_qm54,2026cc_qm17)
  python scripts/setup.py --cuda        # install the CUDA build of PyTorch (NVIDIA GPU)
  python scripts/setup.py --skip-env    # data only, use the Python environment you already have

Needs only the Python standard library (3.10-3.12) and is safe to re-run: finished downloads are skipped and
interrupted ones resume. Data packs come from the GitHub release named by --tag; match videos come from YouTube
(the links The Blue Alliance lists for each match), the same source the pipeline uses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "bruce-peters/frc-auto-scout"
TAG = "data-v1"
TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu128"
DL = ROOT / "data" / "downloads"


def say(msg: str) -> None:
    print(msg, flush=True)


def venv_python() -> Path:
    return ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def run(cmd: list, **kw) -> None:
    say("  $ " + " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, cwd=ROOT, **kw)


# ---------------------------------------------------------------- environment
def setup_env(cuda: bool) -> Path:
    if not (3, 10) <= sys.version_info[:2] <= (3, 12):
        say(f"! Python {sys.version.split()[0]}: auto-scout is tested on 3.10-3.12; continuing anyway")
    py = venv_python()
    if not py.exists():
        say("Creating .venv")
        run([sys.executable, "-m", "venv", ROOT / ".venv"])
    run([py, "-m", "pip", "install", "--upgrade", "pip", "-q"])
    if cuda:
        say("Installing the CUDA build of torch / torchvision")
        run([py, "-m", "pip", "install", "torch", "torchvision", "--index-url", TORCH_CUDA_INDEX, "-q"])
    say("Installing requirements.txt (a few minutes the first time)")
    run([py, "-m", "pip", "install", "-r", ROOT / "requirements.txt", "-q"])
    env, example = ROOT / ".env", ROOT / ".env.example"
    if not env.exists() and example.exists():
        shutil.copyfile(example, env)
        say("Wrote .env from .env.example (add a TBA_API_KEY there to process matches beyond the data pack)")
    return py


# ---------------------------------------------------------------- downloads
def fetch(url: str, dest: Path, size: int | None = None) -> Path:
    """Download url to dest, resuming a partial file; local paths are copied."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not url.startswith(("http://", "https://")):
        shutil.copyfile(url, dest)
        return dest
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    if size and have >= size:
        part.replace(dest)
        return dest
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "auto-scout-setup",
                                                       **({"Range": f"bytes={have}-"} if have else {})})
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and r.status != 206:  # server ignored the range: start over
                    have = 0
                total = size or (have + int(r.headers.get("Content-Length") or 0))
                with open(part, "ab" if have else "wb") as fh:
                    last = 0.0
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        fh.write(chunk)
                        have += len(chunk)
                        if time.time() - last > 2 and total:
                            last = time.time()
                            print(f"\r  {dest.name}: {have / 1e6:,.0f} / {total / 1e6:,.0f} MB", end="", flush=True)
            print(f"\r  {dest.name}: {have / 1e6:,.0f} MB done" + " " * 20, flush=True)
            part.replace(dest)
            return dest
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            have = part.stat().st_size if part.exists() else 0
            say(f"\n  ! {e}; retrying ({attempt + 1}/5)")
            time.sleep(3 * (attempt + 1))
    raise SystemExit(f"Could not download {url}")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def install_packs(base: str, tag: str, want: set[str]) -> dict:
    say(f"Data packs: {', '.join(sorted(want))} (release {tag})")
    man_path = fetch(f"{base}/manifest.json", DL / tag / "manifest.json")
    manifest = json.loads(man_path.read_text())
    for a in manifest["assets"]:
        if a["pack"] not in want:
            continue
        done = DL / tag / (a["name"] + ".done")
        if done.exists():
            say(f"  {a['name']}: already installed")
            continue
        z = fetch(f"{base}/{a['name']}", DL / tag / a["name"], a["size"])
        if sha256(z) != a["sha256"]:
            z.unlink()
            raise SystemExit(f"{a['name']} is corrupt (checksum mismatch); run setup again to re-download it")
        say(f"  extracting {a['name']}")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(ROOT)
        z.unlink()  # extracted: the zip is no longer needed
        done.write_text(a["sha256"])
    return manifest


def download_videos(py: Path, manifest: dict, which: str) -> None:
    videos = manifest.get("videos", {})
    keys = list(videos)
    if which != "all":
        keys = keys[:int(which)] if which.isdigit() else [k.strip() for k in which.split(",") if k.strip()]
    say(f"Match videos: {len(keys)} from YouTube into data/videos/ (~170 MB each)")
    code = ("import sys\nfrom autoscout.video import download, video_path\n"
            "k, yt = sys.argv[1], sys.argv[2]\n"
            "p = video_path(k)\n"
            "print('  have' if p.exists() and p.stat().st_size > 1_000_000 else '  get ', k, flush=True)\n"
            "download(k, yt)\n")
    failed = []
    for k in keys:
        if k not in videos:
            say(f"  ! {k}: not in the data pack's match list")
            continue
        try:
            subprocess.run([str(py), "-c", code, k, videos[k]], check=True, cwd=ROOT)
        except subprocess.CalledProcessError:
            failed.append(k)
    if failed:
        say(f"  ! {len(failed)} videos failed (re-run setup to retry): {', '.join(failed)}")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true", help="also cached scans, 60 Hz fuel logs and all match videos")
    ap.add_argument("--videos", default=None, help="'all', a number, or comma-separated match keys")
    ap.add_argument("--cuda", action="store_true", help="install the CUDA build of PyTorch")
    ap.add_argument("--skip-env", action="store_true", help="do not create .venv or install packages")
    ap.add_argument("--skip-data", action="store_true", help="environment only")
    ap.add_argument("--tag", default=TAG, help=f"data release tag (default {TAG})")
    ap.add_argument("--from", dest="src", default=None, help="a local folder with the data packs instead of GitHub")
    a = ap.parse_args()

    py = Path(sys.executable) if a.skip_env else setup_env(a.cuda)
    if not a.skip_data:
        base = str(Path(a.src).resolve()) if a.src else f"https://github.com/{REPO}/releases/download/{a.tag}"
        manifest = install_packs(base, a.tag, {"starter", "full"} if a.full else {"starter"})
        which = "all" if a.full and a.videos is None else a.videos
        if which:
            download_videos(py, manifest, which)
    run([py, "-c", "import autoscout.pipeline, autoscout.inspector; print('  auto-scout imports OK')"])
    rel = os.path.relpath(py, ROOT)
    say(f"""
Done. Next:
  {rel} -m autoscout.inspector --open        browse every match (video plays once it is in data/videos/)
  {rel} scripts/setup.py --videos 2026cc_qm54 fetch one match's video
  {rel} -m pytest -q tests                   run the tests
  {rel} -m autoscout 2026cc_qm54             re-run a whole match from its video (needs the video; GPU helps)
See README.md ("Things to try") for re-analysing matches with different settings.""")


if __name__ == "__main__":
    main()
