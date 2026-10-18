"""Train an ultralytics YOLO detector. Windows needs the __main__ guard for dataloader workers."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolo26s.pt")
    ap.add_argument("--name", required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--extra", default="", help="k=v,k=v extra train args")
    a = ap.parse_args()
    from ultralytics import YOLO

    try:
        model = YOLO(a.model)
    except Exception as e:  # model name unknown in this ultralytics version
        print("fallback to yolo11s.pt:", e)
        model = YOLO("yolo11s.pt")
    extra = {}
    for kv in filter(None, a.extra.split(",")):
        k, v = kv.split("=")
        try:
            v = float(v) if "." in v else int(v)
        except ValueError:
            pass
        extra[k] = v
    model.train(data=a.data, imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, workers=a.workers,
                project=str(Path(__file__).resolve().parent.parent / "runs"), name=a.name,
                patience=a.patience, exist_ok=True, plots=True, **extra)


if __name__ == "__main__":
    main()
