# FAQ / Troubleshooting

Common errors when using dataflow-cv, and how to fix them.

## "Cannot auto-detect the label format"

Format auto-detection inspects the path you pass: a single `.json` → COCO; a
directory of `.txt` files → YOLO; a directory of `.json` files (first checked
for `shapes` → LabelMe, `images` → COCO).

- Mixed extensions or an empty directory cannot be detected.
- Make sure you point at **labels** (e.g. `yolo_labels/`), not images.
- Multiple paths passed to `analyse stats` must share the same format.

## "Image not found" warnings

Labels and images are matched by file stem — the label file **must share the
image's file name** (`000001.txt` ↔ `000001.jpg`).

- Exception: **LabelMe** files with valid `imageWidth`/`imageHeight` do not
  require the image file (dimensions are read from the JSON).

## "classes.txt is required"

`yolo2coco`, `yolo2labelme`, `labelme2yolo`, `labelme2coco`, and
`visualize yolo` need a class file (one class name per line; the line number
is the class id).

- Converting **from** COCO first? `coco2yolo` / `coco2labelme` auto-generate
  `classes.txt` next to the output; pass `-c` to control the order.

## Validation errors: strict vs non-strict

By default (`strict_mode=True` / no flag), invalid annotations raise errors
and abort. Pass `--no-strict` (CLI) or `strict_mode=False` (API) to skip
invalid files/lines and continue — problems are then reported as warnings.
(`analyse` is always non-strict.)

## "No module named pycocotools"

pycocotools is required for RLE conversion and evaluation:

```bash
pip install dataflow-cv[coco]
```

## RLE conversions crash with UnicodeDecodeError

RLE `counts` bytes must use **latin1**, never UTF-8. This is handled
internally — if you hit a decode error you are likely feeding UTF-8-encoded
RLE data from an external tool; re-save it with latin1 or use polygon
segmentation instead.

## Pose (keypoint) data is dropped / rows are rejected

Pose processing is **explicit** — token counts cannot distinguish a YOLO pose
row from a segmentation row safely:

- Convert: add `--pose` (all six directions); `--kpts-config kpts.yaml`
  supplies keypoint names/skeleton.
- Visualize: `--pose` for YOLO (LabelMe/COCO are automatic); `--kpts-config`
  adds skeleton lines.
- Analyse: `--kpts-config` adds keypoint statistics (and enables YOLO
  pose-row parsing).
- Without the flags, keypoints are dropped with a warning (or YOLO pose rows
  are misparsed as segmentation / rejected).

## YOLO predictions are misparsed or rejected

Prediction files carry an extra `confidence` token (6 tokens for detection
vs 5 for labels). Use `--prediction` with `yolo2coco` — it outputs a plain
JSON list (the standard DT format for `loadRes()`).

## COCO→YOLO / COCO→LabelMe: the `images/` output directory is empty

By design — the converter names outputs by the image file stem and leaves the
`images/` directory for you to fill. Images are never copied.

## Some COCO annotations disappear in conversion

`iscrowd=1` (crowd) annotations are skipped by default on
`coco2yolo` / `coco2labelme`. Pass `--save-crowd` to write them as ordinary
rows/shapes.

## The visualization window won't close

The window-manager close button (X) may not work — always press `q` or `ESC`.
Keyboard controls: `←`/`↑` previous, `→`/`↓`/`Enter`/`Space` next, `s`
snapshot, `h` hints.

## mAP and P/R/F1 together?

They are mutually exclusive in one run: `--prf1` computes single-threshold
P/R/F1 only (fast), without `--prf1` you get the full COCOeval mAP pipeline.
Run the command twice to get both.
