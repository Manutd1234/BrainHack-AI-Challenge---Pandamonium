"""Pre-download CV assets for offline Docker/runtime use."""

from __future__ import annotations

import argparse
import os

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=os.getenv("YOLO_FALLBACK_MODEL", "yolo26l.pt"),
        help="YOLO26 checkpoint name or local path to download/load.",
    )
    args = parser.parse_args()
    YOLO(args.model)
    print(f"CV model asset ready: {args.model}")


if __name__ == "__main__":
    main()
