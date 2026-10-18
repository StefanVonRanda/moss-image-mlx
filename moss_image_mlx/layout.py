"""Helpers for MOSS-Image layout prompts (JSON passed as the prompt text).

MOSS-Image reads boxes as `bbox_format: "yxyx"` in `bbox_coord_space: "abs_centralized"`: absolute pixels with the
canvas centre as origin (x right, y down), so a 1024x1024 canvas spans -512..512 on both axes. These helpers
convert from the conventions people usually have, check a layout against the canvas you are about to render, and
draw the boxes so a layout can be checked before spending minutes on a render.

    moss-mlx-layout check poster.json --size 1024x1024 --preview boxes.png
"""

from __future__ import annotations

import argparse
import json
import sys


def from_top_left(x0: float, y0: float, x1: float, y1: float, width: int, height: int) -> list[int]:
    """Top-left-origin pixel box (x0, y0, x1, y1) -> MOSS centred [y_min, x_min, y_max, x_max]."""
    return [round(y0 - height / 2), round(x0 - width / 2), round(y1 - height / 2), round(x1 - width / 2)]


def from_normalized_1000(box_yxyx: list[float], width: int, height: int) -> list[int]:
    """[y_min, x_min, y_max, x_max] on a 0-1000 grid (common in VLM output) -> MOSS centred pixels."""
    y0, x0, y1, x1 = box_yxyx
    return from_top_left(x0 * width / 1000, y0 * height / 1000, x1 * width / 1000, y1 * height / 1000, width, height)


def load(layout) -> dict:
    return json.loads(layout) if isinstance(layout, str) else layout


def check(layout, width: int, height: int) -> list[str]:
    """Problems that would make the render ignore or misplace elements. Empty list = looks fine."""
    problems = []
    try:
        data = load(layout)
    except json.JSONDecodeError as e:
        return [f"not valid JSON: {e}"]
    if data.get("bbox_format", "yxyx") != "yxyx":
        problems.append("bbox_format should be 'yxyx'")
    if data.get("bbox_coord_space", "abs_centralized") != "abs_centralized":
        problems.append("bbox_coord_space should be 'abs_centralized'")
    if not isinstance(data.get("summary", {}).get("caption", ""), str) or not data.get("summary", {}).get("caption"):
        problems.append("summary.caption is missing: describe the overall scene and style")
    hw, hh = width / 2, height / 2
    for i, item in enumerate(data.get("items", [])):
        name = item.get("text") or item.get("caption", f"item {i}")[:30]
        box = item.get("bbox")
        if not (isinstance(box, list) and len(box) == 4):
            problems.append(f"{name!r}: bbox must be [y_min, x_min, y_max, x_max]")
            continue
        y0, x0, y1, x1 = box
        if y0 >= y1 or x0 >= x1:
            problems.append(f"{name!r}: bbox has min >= max (is it xyxy instead of yxyx?)")
        if min(y0, y1) < -hh - 1 or max(y0, y1) > hh + 1 or min(x0, x1) < -hw - 1 or max(x0, x1) > hw + 1:
            problems.append(f"{name!r}: bbox {box} leaves a {width}x{height} canvas (centred coordinates span "
                            f"x {-hw:.0f}..{hw:.0f}, y {-hh:.0f}..{hh:.0f})")
        if item.get("label") == "text" and not item.get("text"):
            problems.append(f"{name!r}: label 'text' needs a 'text' field with the exact string to render")
    if not data.get("items"):
        problems.append("no items")
    return problems


def preview(layout, width: int, height: int, path: str):
    """Draw the layout's boxes and texts on a blank canvas."""
    from PIL import Image, ImageDraw

    data = load(layout)
    img = Image.new("RGB", (width, height), (245, 243, 238))
    d = ImageDraw.Draw(img)
    colours = {"text": (40, 90, 200), "object": (220, 110, 30)}
    for item in data.get("items", []):
        y0, x0, y1, x1 = item["bbox"]
        box = [x0 + width / 2, y0 + height / 2, x1 + width / 2, y1 + height / 2]
        c = colours.get(item.get("label"), (90, 90, 90))
        d.rectangle(box, outline=c, width=3)
        d.text((box[0] + 6, box[1] + 4), item.get("text") or item.get("caption", "")[:60], fill=c)
    img.save(path)


def main(argv=None):
    p = argparse.ArgumentParser(prog="moss-mlx-layout", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["check"])
    p.add_argument("layout", help="layout JSON file")
    p.add_argument("--size", default="1024x1024", help="canvas WxH you will render at")
    p.add_argument("--preview", help="write a PNG with the boxes drawn")
    a = p.parse_args(argv)
    w, h = (int(v) for v in a.size.lower().split("x"))
    text = open(a.layout, encoding="utf-8").read()
    problems = check(text, w, h)
    for problem in problems:
        print("warning:", problem)
    if a.preview:
        preview(text, w, h, a.preview)
        print(a.preview)
    if not problems:
        print(f"ok: {len(load(text).get('items', []))} items fit a {w}x{h} canvas")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
