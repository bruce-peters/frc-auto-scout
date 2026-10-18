"""python -m autoscout.inspector [--port 8765] [--open]"""
import argparse

from .server import serve


def main():
    ap = argparse.ArgumentParser(prog="python -m autoscout.inspector")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--open", action="store_true", help="open the browser")
    a = ap.parse_args()
    serve(a.port, a.open)


if __name__ == "__main__":
    main()
