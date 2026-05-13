"""Pre-download the YOLO26 surrogate used by the noise attack."""

from __future__ import annotations

import argparse
import os

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=os.getenv("YOLO_SURROGATE_FALLBACK", "yolo26l.pt"),
    )
    args = parser.parse_args()
    YOLO(args.model)
    print(f"Noise surrogate asset ready: {args.model}")


if __name__ == "__main__":
    main()
