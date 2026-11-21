#!/usr/bin/env python3
"""README showcase image generator.

Renders detection, segmentation, and pose visualizations for the README
showcase section (``assets/showcase/det_demo.jpg``, ``seg_demo_*.jpg``,
``pose_demo.jpg``) from real COCO val2017 images.  The script filters COCO
JSONs down to the selected images, renders them headlessly via the COCO
visualizer, and optionally copies the results into ``assets/showcase/``.

Example::

    python samples/visualize/showcase_demo.py \\
        --coco-json ~/datasets/coco/annotations/instances_val2017.json \\
        --keypoints-json ~/datasets/coco/annotations/person_keypoints_val2017.json \\
        --image-dir ~/datasets/coco/images/val2017 \\
        --out /tmp/showcase --write-showcase

Image selection: default ids are chosen for category diversity and
medium-sized objects (good mask visibility); the pose image has a fully
labeled (17/17) person.  Override with ``--ids`` / ``--pose-ids``.
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
DET_IDS = [554291]  # cat + dog living-room scene (det_demo.jpg)
SEG_IDS = [357459, 329447]  # two diverse scenes (seg_demo_*.jpg)
POSE_IDS = [368212]  # single person, 17/17 keypoints (pose_demo.jpg)


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


def render(annotation_file: Path, image_dir: Path, out: Path) -> list[Path]:
    """Render a filtered COCO JSON headlessly; return the newly rendered jpgs."""
    before = set(out.glob("*_visualized.jpg"))
    visualizer = COCOVisualizer(
        annotation_file=str(annotation_file),
        image_dir=str(image_dir),
        is_show=False,
        is_save=True,
        output_dir=str(out),
        log_config=LogConfig(name="showcase"),
    )
    result = visualizer.visualize()
    if not result.success:
        raise SystemExit(f"Visualization failed: {result.errors}")
    new = sorted(set(out.glob("*_visualized.jpg")) - before)
    if not new:
        raise SystemExit(f"No renders produced for {annotation_file}")
    return new


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coco-json",
        type=Path,
        required=True,
        help="COCO annotation JSON (e.g. instances_val2017.json)",
    )
    parser.add_argument(
        "--keypoints-json",
        type=Path,
        default=None,
        help="COCO keypoints JSON (e.g. person_keypoints_val2017.json) for the pose demo",
    )
    parser.add_argument(
        "--image-dir", type=Path, required=True, help="Directory containing the image files"
    )
    parser.add_argument(
        "--ids",
        type=int,
        nargs="+",
        default=DET_IDS + SEG_IDS,
        help="Image ids for det/seg demos (default: one det + two seg scenes)",
    )
    parser.add_argument(
        "--pose-ids",
        type=int,
        nargs="+",
        default=POSE_IDS,
        help="Image ids for the pose demo",
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
        help="Copy renders into assets/showcase/",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    renders: dict[str, Path] = {}

    # Detection demo — first of --ids.
    det_filtered = args.out / "_showcase_det.json"
    filter_coco_json(args.coco_json, args.ids[:1], det_filtered)
    renders["det_demo"] = render(det_filtered, args.image_dir, args.out)[0]

    # Segmentation demos — the remaining --ids (rendered in one pass).
    seg_filtered = args.out / "_showcase_seg.json"
    filter_coco_json(args.coco_json, args.ids[1:], seg_filtered)
    renders.update(
        {
            f"seg_demo_{i}": p
            for i, p in enumerate(render(seg_filtered, args.image_dir, args.out), start=1)
        }
    )

    # Pose demo — from the keypoints JSON (skeleton comes from categories).
    if args.keypoints_json is not None:
        pose_filtered = args.out / "_showcase_pose.json"
        filter_coco_json(args.keypoints_json, args.pose_ids, pose_filtered)
        renders["pose_demo"] = render(pose_filtered, args.image_dir, args.out)[0]

    if args.write_showcase:
        showcase_dir = Path(__file__).parent.parent.parent / "assets" / "showcase"
        for name, render_path in renders.items():
            shutil.copy2(render_path, showcase_dir / f"{name}.jpg")
        print(f"Copied {len(renders)} renders to {showcase_dir}: {sorted(renders)}")


if __name__ == "__main__":
    main()
