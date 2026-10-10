#!/usr/bin/env python3
"""README showcase image generator.

Renders segmentation visualizations for the README showcase section
(``assets/showcase/seg_demo_*.jpg``) from real COCO images.  The script
filters a COCO JSON down to the selected images, renders them headlessly
via the COCO visualizer, and optionally copies the results into
``assets/showcase/``.

Example::

    python samples/visualize/showcase_demo.py \\
        --coco-json ~/datasets/coco/annotations/instances_val2017.json \\
        --image-dir ~/datasets/coco/images/val2017 \\
        --out /tmp/showcase --write-showcase

Image selection: default ids are chosen for category diversity and
medium-sized objects (good mask visibility).  Override with ``--ids``.
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from dataflow.visualize import COCOVisualizer
from dataflow.util import LogConfig

# Default showcase images (COCO val2017) — clean scenes with few
# medium-sized objects, selected for sharp/colorful imagery.
DEFAULT_IDS = [357459, 329447]


def filter_coco_json(coco_path: Path, ids: list[int], out_path: Path) -> None:
    """Write a COCO JSON containing only the selected images."""
    data = json.loads(coco_path.read_text(encoding="utf-8"))
    idset = set(ids)
    images = [im for im in data["images"] if im["id"] in idset]
    anns = [a for a in data["annotations"] if a["image_id"] in idset and a.get("iscrowd", 0) == 0]
    cat_ids = {a["category_id"] for a in anns}
    cats = [c for c in data["categories"] if c["id"] in cat_ids]
    out_path.write_text(
        json.dumps(
            {
                "info": {"description": "showcase subset"},
                "images": images,
                "annotations": anns,
                "categories": cats,
            }
        ),
        encoding="utf-8",
    )
    print(f"Filtered {len(images)} images, {len(anns)} annotations, {len(cats)} categories")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coco-json",
        type=Path,
        required=True,
        help="COCO annotation JSON (e.g. instances_val2017.json)",
    )
    parser.add_argument(
        "--image-dir", type=Path, required=True, help="Directory containing the image files"
    )
    parser.add_argument(
        "--ids",
        type=int,
        nargs="+",
        default=DEFAULT_IDS,
        help="Image ids to render (default: two diverse scenes)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(tempfile.mkdtemp()),
        help="Output directory for rendered images",
    )
    parser.add_argument(
        "--write-showcase",
        action="store_true",
        help="Copy renders into assets/showcase/ as seg_demo_*.jpg",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    filtered = args.out / "_showcase_anns.json"
    filter_coco_json(args.coco_json, args.ids, filtered)

    visualizer = COCOVisualizer(
        annotation_file=str(filtered),
        image_dir=str(args.image_dir),
        is_show=False,
        is_save=True,
        output_dir=str(args.out),
        log_config=LogConfig(name="showcase"),
    )
    result = visualizer.visualize()
    if not result.success:
        raise SystemExit(f"Visualization failed: {result.errors}")

    if args.write_showcase:
        showcase_dir = Path(__file__).parent.parent.parent / "assets" / "showcase"
        renders = sorted(args.out.glob("*_visualized.jpg"))
        if not renders:
            raise SystemExit("No renders produced")
        for i, render in enumerate(renders, start=1):
            shutil.copy2(render, showcase_dir / f"seg_demo_{i}.jpg")
        print(f"Copied {len(renders)} renders to {showcase_dir}")


if __name__ == "__main__":
    main()
