"""Build the downloadable data packs that scripts/setup.py installs (maintainers only).

  python scripts/pack_data.py [--tag data-v1] [--out dist]

Writes dist/<tag>/:
  starter.zip      every processed match's results (output/<match>/ without scan.pkl), the trained models,
                   the TBA / calibration / overlay caches and the DINO identity probes: enough to browse every
                   match in the inspector and to process new matches
  scans-NN.zip     output/<match>/scan.pkl, the cached detections each re-analysis starts from (full pack)
  fuel60-NN.zip    60 Hz fuel blob logs of the true-60p webcasts (data/exp_track/blobs60, full pack)
  manifest.json    name, pack, size and sha256 of every zip, plus the match list and YouTube keys

Each zip holds paths relative to the repository root and stays under GitHub's 2 GB release-asset limit.
Match videos are not packed: setup.py --full downloads them from YouTube, the source the pipeline uses.
5026's scouting export (data/scouting/) is never packed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from autoscout import config, tba  # noqa: E402

PART_MAX = 1_850_000_000  # bytes of input per zip (GitHub release assets must be < 2 GiB)
MODELS = ("robots.pt", "fuel.pt")  # the detectors the pipeline loads (tracked JSON / npz / pkl models come with git)
OUTPUT_ROOT_FILES = ("evaluation.json", "identity_holdout.json", "attribution_tuning.json", "vision_counter_eval.json",
                     "index.html")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_zip(path: Path, files: list[Path], level: int | None) -> None:
    comp = zipfile.ZIP_STORED if level is None else zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(path, "w", compression=comp, compresslevel=level, allowZip64=True) as z:
        for f in files:
            z.write(f, f.relative_to(ROOT).as_posix())
    print(f"  {path.name}: {len(files)} files, {path.stat().st_size / 1e9:.2f} GB", flush=True)


def parts(files: list[Path]) -> list[list[Path]]:
    out, cur, size = [], [], 0
    for f in files:
        s = f.stat().st_size
        if cur and size + s > PART_MAX:
            out.append(cur)
            cur, size = [], 0
        cur.append(f)
        size += s
    if cur:
        out.append(cur)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="data-v1")
    ap.add_argument("--out", default=str(ROOT / "dist"))
    ap.add_argument("--skip-full", action="store_true", help="starter pack only")
    a = ap.parse_args()
    out = Path(a.out) / a.tag
    out.mkdir(parents=True, exist_ok=True)
    keys = sorted(p.parent.name for p in config.OUTPUT.glob("2026*/result.json"))
    print(f"{len(keys)} matches -> {out}", flush=True)

    starter = [config.MODELS / m for m in MODELS if (config.MODELS / m).exists()]
    starter += [f for f in sorted(config.CACHE.rglob("*")) if f.is_file()]
    starter += [f for f in sorted((config.DATA / "exp_dino" / "probe" / "causal").glob("*.npz"))]
    starter += [config.OUTPUT / n for n in OUTPUT_ROOT_FILES if (config.OUTPUT / n).exists()]
    for k in keys:
        starter += [f for f in sorted((config.OUTPUT / k).iterdir())
                    if f.is_file() and f.name != "scan.pkl" and f.suffix != ".mp4"]
    write_zip(out / "starter.zip", starter, 6)
    assets = [("starter.zip", "starter")]

    if not a.skip_full:
        scans = [config.OUTPUT / k / "scan.pkl" for k in keys if (config.OUTPUT / k / "scan.pkl").exists()]
        for i, grp in enumerate(parts(scans), 1):
            write_zip(out / f"scans-{i:02d}.zip", grp, 1)
            assets.append((f"scans-{i:02d}.zip", "full"))
        logs = [Path(config.BLOBS60) / f"{k}.pkl.gz" for k in keys]
        logs = [p for p in logs if p.exists()]
        # 60 Hz logs of webcasts that show each 30p picture twice add nothing (autoscout/tracks60.py): leave them out
        true60 = config.DATA / "exp_track" / "true60p.txt"
        if true60.exists():
            keep = set(true60.read_text().split())
            logs = [p for p in logs if p.name.split(".")[0] in keep]
        for i, grp in enumerate(parts(logs), 1):
            write_zip(out / f"fuel60-{i:02d}.zip", grp, None)
            assets.append((f"fuel60-{i:02d}.zip", "full"))

    videos = {}
    for k in keys:
        try:
            yt = tba.match(k).youtube
            if yt:
                videos[k] = yt[0]
        except Exception as e:  # noqa: BLE001
            print("  no video key for", k, e)
    manifest = {"tag": a.tag, "matches": keys, "videos": videos,
                "assets": [{"name": n, "pack": p, "size": (out / n).stat().st_size, "sha256": sha256(out / n)}
                           for n, p in assets]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    tot = {p: sum(x["size"] for x in manifest["assets"] if x["pack"] == p) for p in ("starter", "full")}
    print(f"manifest.json: starter {tot['starter'] / 1e9:.2f} GB, full {tot['full'] / 1e9:.2f} GB, "
          f"{len(videos)} video keys", flush=True)


if __name__ == "__main__":
    main()
