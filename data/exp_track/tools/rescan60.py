"""Fuel-only re-scan at 60 Hz (Options.fuel_step = 1) of a few matches: blob logs to data/exp_track/blobs60/.
Reads output/<key>/scan.pkl (cams, layout, timeline) read-only; nothing under output/ is written."""
import gzip, os, pickle, sys, time
from pathlib import Path
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from autoscout import config, pipeline  # noqa: E402
from autoscout.video import Video, video_path  # noqa: E402
out = config.DATA / "exp_track" / "blobs60"
out.mkdir(parents=True, exist_ok=True)
for key in sys.argv[1:]:
    dst = out / f"{key}.pkl.gz"
    lock = out / f"{key}.lock"
    if dst.exists() or (out / f"{key}.pkl").exists():
        continue
    try:  # a second rescan process skips matches the first one is on
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        continue
    tic = time.time()
    with open(config.OUTPUT / key / "scan.pkl", "rb") as fh:
        md = pickle.load(fh)
    opts = pipeline.Options(fuel_step=1)
    fs = pipeline.rescan_fuel(md, Video(video_path(key)), opts, keep_blobs=True)
    cpu = time.process_time()
    with gzip.open(dst, "wb", compresslevel=3) as fh:
        pickle.dump({"fuel_step": 1, "blob_log": fs.blob_log}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    lock.unlink(missing_ok=True)
    print(f"{key}: {time.time() - tic:.0f}s wall, {cpu:.0f}s cpu, {sum(len(r) for r in fs.blob_log.values())} centre frames", flush=True)
