#!/usr/bin/env python3
"""Interactively collect fixed-L515 RGB images: K saves one frame, Q exits."""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from box_perception.camera.rgb_capture import next_capture_index, save_rgb_png  # noqa: E402
from box_perception.camera.ros_color import ROSColorSource  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config/camera.yaml")
    parser.add_argument("--color-topic", help="override config ros.color_topic")
    parser.add_argument("--prefix", default="rgb")
    parser.add_argument("--timeout", type=float, default=12.0)
    parser.add_argument("--display-scale", type=float, default=0.75)
    parser.add_argument("--min-save-interval", type=float, default=0.25)
    args = parser.parse_args()
    if args.timeout <= 0.0 or args.display_scale <= 0.0 or args.min_save_interval < 0.0:
        parser.error("timeout/display scale must be positive; save interval cannot be negative")
    if not args.prefix or any(character in args.prefix for character in "/\\"):
        parser.error("--prefix must be a non-empty filename prefix without path separators")
    return args


def _annotated_preview(image, *, saved: int, next_index: int, fps: float, output: Path):
    preview = image.copy()
    lines = [
        "K: save one RGB frame    Q: finish and quit",
        f"saved={saved}  next={next_index:06d}  display={fps:.1f} FPS",
        f"output={output}",
    ]
    for index, line in enumerate(lines):
        y = 38 + index * 34
        cv2.putText(preview, line, (22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 0, 0), 5)
        cv2.putText(preview, line, (22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 255, 0), 2)
    return preview


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    color_topic = args.color_topic or str(config["ros"]["color_topic"])
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    next_index = next_capture_index(output, args.prefix)
    initial_index = next_index
    latest = None
    last_saved_sequence = -1
    last_save_time = -float("inf")
    arrival_times: deque[float] = deque(maxlen=31)
    last_sequence = -1
    window_name = "Fixed L515 YOLO RGB Collector"

    print("========== fixed L515 interactive RGB collection ==========")
    print(f"topic: {color_topic}")
    print(f"output: {output}")
    print(f"next file: {args.prefix}_{next_index:06d}.png")
    print("Focus the image window. Press K to save one raw RGB frame; press Q to finish.")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        with ROSColorSource(color_topic) as source:
            latest = source.read(args.timeout)
            while True:
                frame = source.poll(0.001)
                if frame is not None:
                    latest = frame
                    if frame.sequence != last_sequence:
                        arrival_times.append(time.monotonic())
                        last_sequence = frame.sequence
                fps = 0.0
                if len(arrival_times) >= 2:
                    elapsed = arrival_times[-1] - arrival_times[0]
                    fps = (len(arrival_times) - 1) / elapsed if elapsed > 0.0 else 0.0
                preview = _annotated_preview(
                    latest.image_bgr,
                    saved=next_index - initial_index,
                    next_index=next_index,
                    fps=fps,
                    output=output,
                )
                if args.display_scale != 1.0:
                    preview = cv2.resize(
                        preview,
                        None,
                        fx=args.display_scale,
                        fy=args.display_scale,
                        interpolation=cv2.INTER_AREA,
                    )
                cv2.imshow(window_name, preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break
                if key in (ord("k"), ord("K")):
                    now = time.monotonic()
                    if latest.sequence == last_saved_sequence:
                        print("skip: no new RGB frame since the previous save")
                    elif now - last_save_time < args.min_save_interval:
                        print("skip: K repeat was faster than --min-save-interval")
                    else:
                        path = save_rgb_png(output, latest.image_bgr, next_index, args.prefix)
                        print(f"saved {next_index - initial_index + 1}: {path.name}")
                        last_saved_sequence = latest.sequence
                        last_save_time = now
                        next_index += 1
    except KeyboardInterrupt:
        print("Ctrl+C received; finishing the current collection session")
    finally:
        cv2.destroyAllWindows()

    print(f"Capture complete: saved {next_index - initial_index} new RGB images to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
