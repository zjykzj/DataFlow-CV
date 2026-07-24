# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.9.0] - 2026-07-24

### Added

- **File sampling** (`analyse sample`): New `SampleAnalyser` collects N files from a dataset with three modes (labels-only, images-only, both) and two strategies — random (`--shuffle`, default) or sequential (`--no-shuffle`, first N in sort order). Pure file-level operation with no annotation parsing. Supports `--move` for relocating files and `--class-file` for copying classes.txt to the output. YOLO and LabelMe only (COCO rejected).
- **`_collect_label_files` shared utility**: Extracted from `split.py` to `utils.py` for reuse across `SplitAnalyser`, `SampleAnalyser`, and future analysers.

### Docs

- **Specs**: `spec_analyse.md` v2.1 documents the full SampleAnalyser contract (pipeline, behavior table, output layout, error scenarios). `spec_cli.md` v2.3 adds the `sample` subcommand with option table and mode documentation.
- **README & CLAUDE.md**: Feature table updated with file sampling; CLI quick-start and Python API examples added for `sample`; test count refreshed (535→556). CLAUDE.md Analyse section now covers all five analysers.
- **Samples**: `sample_demo.py` demonstrates all three modes and both strategies with test data.

## [1.8.0] - 2026-07-17

### Added

- **Category filtering** (`analyse filter`): New `FilterAnalyser` keeps only specified classes and remaps class IDs to the new class file order. Supports YOLO (streaming), LabelMe (streaming), and COCO (batch) — same-format only, with automatic format detection.
- **Multi-path stats and recursive traversal**: `analyse stats` now accepts multiple `LABEL_PATH` arguments (statistics merged across paths by class name) and `-R/--recursive` for recursive label discovery in subdirectories. When `--class-file` is provided, class IDs/names outside the class file produce an ERROR instead of being silently dropped.
- **N-way dataset partitioning** (`analyse partition`): New `PartitionAnalyser` splits a dataset into N roughly-equal parts. Three modes: labels-only, images-only, or both (images matched by stem). Supports `--shuffle` with `--seed` for reproducible random distribution and `--move` for storage-constrained scenarios. YOLO and LabelMe only (COCO users are directed to `split`).

### Changed

- **Stats performance**: `StatsAnalyser` skips image loading for YOLO datasets (`skip_image_loading` parameter) — annotation counting no longer calls `cv2.imread()` per label file, reducing stats runtime from minutes to sub-second on typical datasets.
- **Float-tolerant class ID parsing**: `parse_yolo_class_id()` in `dataflow/label/utils.py` is now the canonical parser for YOLO class IDs, accepting both `"5"` and float-formatted `"5.000000"` tokens.

### Fixed

- **Recursive discovery on WSL**: YOLO/LabelMe handlers now support a native `recursive` parameter using `rglob`, replacing the temp-dir + symlink approach that failed on WSL cross-filesystem setups.
- **Code review findings**: Comprehensive review round — conversion result logging now runs in both batch and streaming pipelines, `atexit` cleanup for auto-generated class files, dead code removal (unused exception classes, `RenderData` dimension fields, duplicate validators), and shared-utility extraction (RLE warnings, COCO category pre-loading, classes-file I/O).
- **CLI help text formatting**: Command descriptions no longer carry stray indentation (`inspect.cleandoc()` applied), the partition mode table is no longer rewrapped into garbled text (`\b` placement), and `analyse` subcommands show accurate `OUTPUT_DIR` descriptions.

### Docs

- **SDD workflow**: `/spec` skill now defines the spec-first development loop (update spec before coding, conformance-check after), backed by CLAUDE.md hard rules and a PreToolUse hook that injects the methodology reminder on any `specs/` edit.
- **Skills portability**: Removed project-specific hardcoded paths from `/dev` and `/release`; all skills now document their required configuration and use template variables defined in CLAUDE.md.
- **README sync**: Feature table, CLI quick start, and Python API examples now cover category filtering and N-way partitioning; test stats refreshed (535 tests, 79% coverage) with a rebuilt per-module coverage table.
- **Spec updates**: `spec_analyse.md` documents the full FilterAnalyser/PartitionAnalyser contracts and multi-path stats; `skip_image_loading` and `parse_yolo_class_id()` documented in `spec_label.md`.

## [1.7.0] - 2026-07-12

### Added

- **Analyse module**: New `dataflow.analyse` module for dataset introspection and preparation. Includes `StatsAnalyser` (dataset statistics with per-class counts) and `SplitAnalyser` (train/val split with seed-based shuffling). Supports YOLO, LabelMe, and COCO formats with automatic format detection.
- **Stats sorting options**: `stats` command now supports `--sort-by` (`id`/`count`) and `--descending`/`--ascending` to control per-class table ordering. Added ID column to the per-class statistics table.
- **`--log-dir` for evaluate**: `evaluate` subcommands now accept `--log-dir` to specify log output directory, consistent with other CLI subcommands.

### Changed

- **CLI imports normalized**: CLI commands now import only through package-level public API (`dataflow.convert`, `dataflow.visualize`, `dataflow.evaluate`), respecting the architecture layering constraint.
- **Analyse help text formatting**: Improved `analyse` subcommand `--help` output alignment.

### Fixed

- **Spec-code alignment**: Comprehensive audit fixed inconsistencies between 16 spec files and implementation — coordinate clamping thresholds, error message content, import paths, and validation behavior.
- **Converter state cleanup**: `_source_annotations_for_target` is now properly cleared in a `finally` block on batch conversion to prevent stale state across conversions.
- **Prediction output format**: COCO prediction output now correctly uses Variant B (plain JSON list), matching the format expected by pycocotools `loadRes()`.
- **Evaluate validation**: Input validation now collects all errors before raising, ensuring every validation problem is visible in both logs and programmatic output.
- **YOLO handler image requirement**: YOLO handler no longer requires image files to exist when labels can be read independently.
- **Analyse bugs and spec consistency**: Fixed unused imports, incorrect spec cross-references, and stale parameter documentation.

### Docs

- **Samples and README sync**: Demo scripts and README updated to reflect current API (LogConfig-based logging, CLI command structure).
- **Module ordering unified**: All docs now consistently use Analyse → Convert → Visualize → Evaluate ordering.

## [1.6.2] - 2026-07-02

### Docs

- **Project skills**: Extracted git workflows (`/commit`, `/release`), development workflows (`/dev`), spec maintenance (`/spec`), and CLAUDE.md authoring (`/claude`) from CLAUDE.md into reusable project skills. Skills use placeholder notation for cross-project portability; project-specific config retained in CLAUDE.md.
- **Specs audit and normalization**: Unified version headers (`vX.Y | Last Updated: YYYY-MM-DD`) across all 16 spec files per `/spec` methodology. Removed changelog/migration notes. Fixed section numbering gaps.
- **Spec-code alignment**: Audited all specs against code — added undocumented `method`/`class_names`/`prediction` fields to specs, updated ColorManager algorithm to match golden ratio implementation, marked `strict_mode` as planned.

### Changed

- **CLAUDE.md restructured**: Spec maintenance methodology, development commands, and test structure moved to skills. Added AI model configuration and release configuration sections.

### Fixed

- **Legacy `logger` parameter removed**: Removed deprecated `logger` parameter from `compute_pr_f1()` — no callers were using it.

## [1.6.1] - 2026-07-02

### Docs

- **Specs purge**: Removed non-contract content from all specs — Change History, File Map directory trees, Implementation Steps pseudocode, Migration guides, and Legacy API tables. Specs now contain only behavioral contracts and contract-supporting explanations.
- **CLAUDE.md Spec Maintenance methodology**: Embedded ~55-line methodology covering classification principles, two-reader model (agent + human), six deletion types, modules interface-vs-implementation check, and two edge-case rules (comparison tables, outdated framing).
- **Removed SDD_GUIDE.md and SDD_METHODOLOGY.md from specs/**: Methodology now lives in CLAUDE.md; reusable template available in docs/.

### Fixed

- **CLI pycocotools imports (spec §7 compliance)**: Removed direct `import pycocotools` from CLI; evaluate commands use `_validate_coco_available()` from `dataflow.evaluate.utils`; convert commands delegate RLE availability check to converter internals.
- **CLI error messages**: Fixed `--image-dir`/`--class-file` references in validate_convert_params error messages to use actual positional argument names (`IMAGE_DIR`/`CLASS_FILE`).
- **Converters cross-layer import**: Replaced `from ..label.coco_handler import HAS_COCO_MASK` with local try/except pycocotools check in `yolo_and_coco.py` and `coco_and_labelme.py`.
- **Evaluate CLI exception type**: pycocotools unavailability now correctly raises `SystemError` (exit 5) instead of `ImportError` per spec §5.
- **Spec CLI allowed imports**: Added `dataflow.evaluate.utils` to §7 allowed list.

## [1.6.0] - 2026-06-27

### Added

- **Progress output during conversion**: Convert module now reports real-time progress during batch and streaming conversions, including image count, annotation count, and elapsed time.
- **YOLO normalized coordinate clamping**: Coordinates outside [0,1] in YOLO annotations are now automatically clamped to valid range before validation, with WARNING emitted when change exceeds threshold.

### Changed

- **Clamping precision**: WARNING threshold raised to `1e-6` for normalized coordinates and output precision increased from `.6f` to `.10f` to reduce false-positive warnings from float round-trip noise.

### Docs

- **README tagline**: Replaced internal development references with a product-focused tagline describing what the tool does for users.
- **README & CLAUDE.md sync**: Updated test count (418→440), code coverage statistics, per-module coverage table, added missing `spec_logging.md` to CLAUDE.md spec tree, and fixed `exceptions.py` path.

## [1.5.0] - 2026-06-13

### Added

- **Industry-standard label positioning** in visualization: labels are placed above bbox top-left (not top-center), background uses class color (not black), and edge case flips label inside bbox (matching Ultralytics YOLOv8, Supervision, Detectron2 conventions).
- **Bbox from polygon fallback**: When YOLO segmentation annotations lack an explicit bbox, the visualizer computes axis-aligned bounds from polygon points for consistent label placement.
- **`NaturalOrderGroup`** in CLI: command registration order is preserved in `--help` output (`convert → visualize → evaluate` instead of alphabetical).

### Changed

- **CLAUDE.md**: Added Module Ordering Convention (Convert → Visualize → Evaluate) mirroring the existing Format Ordering Convention.
- **README**: Simplified Evaluation section (merged sections ①+④, removed field-requirement table, added showcase images).
- **Specs** (`spec_visualize.md`): Updated to v4.3 with per-annotation drawing flow, label positioning rules, and change history.

### Fixed

- **CLI `--help` order**: Top-level commands now display as `convert → visualize → evaluate` matching docs.
- **Broken COCO demo**: Removed invalid `label_dir` parameter passed to `COCOVisualizer`.
- **Dead code**: Removed unused `calculate_text_position` from `visualize/utils.py`.

## [1.4.0] - 2026-06-13

### Added

- **Unified `LogManager`** (`dataflow/util/logging.py`): Single entry point replacing `LoggingOperations` and `VerboseLoggingOperations`. `LogConfig` (frozen dataclass with `name`, `verbose`, `log_dir`) passed to all module constructors via `log_config` parameter. Console handler always active (INFO, compact format); file handler (DEBUG, RotatingFileHandler 10MB×5) when `verbose=True`.
- **Per-module log templates**: `dataflow/{convert,visualize,evaluate}/log_templates.py` with structured formatting functions (`format_convert_header`, `format_viz_progress`, `format_metric_table`, etc.).
- **CLI `--log-dir` option**: Configurable log file output directory (default `./logs`) on `convert` and `visualize` commands.
- **`tests/conftest.py`**: Shared session-scoped fixtures (`project_root`, `test_data_dir`, `test_data_det`, `test_data_seg`, `test_data_evaluate`).
- **CLI evaluate tests** (`tests/cli/test_evaluate.py`): 16 tests covering `detection` and `segmentation` subcommands with all flags.
- **`SegmentationEvaluator` success path tests**: 3 tests with real segmentation data (was only tested for failure on bbox-only data).
- **Evaluate samples** (`samples/evaluate/`): `detection_demo.py` and `segmentation_demo.py` with `--prf1` support.

### Changed

- **Logging ownership**: All log output is produced by modules, not CLI. CLI passes `LogConfig` to module constructors and uses `click.echo()` for terminal UI. No duplicate log files.
- **`_log_error` inlined per base class**: Each base class has its own implementation (no shared utility). Label: ERROR + raise in strict mode. Convert: same. Visualize: ERROR (never raises). Evaluate: ERROR + always raise.
- **`--prf1` redesign**: Now computes P/R/F1 only (skips COCOeval entirely). mAP and P/R/F1 are mutually exclusive CLI paths. Run twice for both metrics.
- **Result objects**: `log_file_path` field renamed to `log_path` in `ConversionResult`, `VisualizationResult`, `EvaluationResult`.
- **README redesigned**: Header with format badges + 4-row feature table replacing Mermaid diagram; `Features` section removed (merged into header + Key Concepts); Installation simplified.
- **Test count**: 405 → 418.

### Removed

- **`LoggingOperations` / `VerboseLoggingOperations`** (`dataflow/util/logging_util.py`): Replaced by `LogManager`.
- **`FileOperations`** (`dataflow/util/file_util.py`): Removed — file I/O uses inline `pathlib.Path` methods directly.
- **`logging_error_or_raise()`**: Inlined into each base class's `_log_error`.
- **`samples/label/`**: Handler demos removed (not user entry points).
- **`samples/util/`**: Demos for deleted classes removed.
- **`pylint` / `docs` optional-deps / `package-data`**: Cleaned from `pyproject.toml`.

### Fixed

- **All 10 convert/visualize samples**: Broken by previous bulk script — rewritten with clean `LogConfig` pattern, no `sys.path` hacks.
- **Duplicate log files**: Convert CLI path no longer creates a second logger internally.

## [1.3.0] - 2026-06-12

### Added

- **Macro/micro averaging for P/R/F1**: `compute_pr_f1()` now accepts a `method` parameter (default `"macro"`, optional `"micro"`) to control overall P/R/F1 aggregation. Macro averaging treats each category equally; micro averaging weights by annotation count. The `PRF1Result.method` field records which method was used, and `format_prf1_output()` displays it.
  - CLI: `--prf1-method` option (Choice: `macro|micro`, default `macro`) on `evaluate detection` and `evaluate segmentation` subcommands.
  - Per-class results are identical between modes; only `overall` differs. TP/FP/FN in `overall` are always summed totals.
- **Mask IoU support for segmentation PRF1**: `compute_pr_f1(iou_type='segm')` now works via pycocotools `mask` module (`mask.frPyObjects()` + `mask.merge()` for polygon→RLE, `mask.iou()` for batched computation). Both polygon and RLE input formats are supported. The previous `NotImplementedError` and CLI bbox fallback have been removed.

### Changed

- Specs: `spec_evaluate_metrics.md` v1.1 (macro/micro formulas + comparison), `spec_evaluate.md` v1.2 (mask IoU algorithm), `spec_cli.md` v1.1 (`--prf1-method` option).
- Test count: 397 → 405.

## [1.2.0] - 2026-06-11

### Added

- **Streaming visualization pipeline**: Redesigned `Visualize` module to stream images one at a time via `handler.iter_images()` instead of batch-loading all annotations upfront. First image appears as soon as the first annotation file is parsed, with low memory usage (single image at a time).
  - `BaseAnnotationHandler.iter_images()`: New abstract method yielding `ImageAnnotation` one at a time. Implemented in YOLO, COCO, and LabelMe handlers.
  - `BaseVisualizer._create_handler()` and `_convert_to_render_data()`: Replace `load_annotations()`. Per-image coordinate conversion and rendering.
  - Counter-based progress display replaces percentage bars (total unknown in streaming mode).
- **Streaming conversion pipeline**: `Convert` module now auto-selects batch vs streaming based on target format.
  - `BaseConverter.stream_convert()`: Template method using `iter_images()` + `_convert_single_image()` + `write_one()`.
  - `BaseConverter._convert_single_image()`: Abstract method — single source of truth for per-image coordinate transforms. `convert_annotations()` delegates to it in a loop.
  - Streaming directions: COCO→YOLO, LabelMe↔YOLO, COCO→LabelMe (per-file output).
  - Batch directions: YOLO→COCO, LabelMe→COCO (single JSON output — required by COCO format).
- **Per-image write API**: `YoloAnnotationHandler.write_one()` and `LabelMeAnnotationHandler.write_one()` (renamed from `_write_single_image`).
- **Category extraction for streaming**: `_ensure_categories_for_streaming()` pre-loads categories from COCO JSON before streaming iteration, enabling `classes.txt` generation in the target handler.

### Changed

- Visualize pipeline: `load_annotations()` (batch) → `_create_handler()` + `iter_images()` + `_convert_to_render_data()` (streaming)
- Convert pipeline: `convert_annotations()` now delegates to `_convert_single_image()` per image.
- Label handler: `_write_single_image()` renamed to public `write_one()` on YOLO and LabelMe handlers.
- Progress display: Counter format (`Processed 40 images, 0 failed`) replaces ASCII percentage bar in visualize.
- Test count: 362 → 370.

### Documentation

- `specs/modules/spec_label.md` v3.0: Added `iter_images()` streaming iterator contract.
- `specs/modules/spec_visualize.md` v3.0: Redesigned pipeline to streaming per-image.
- `specs/modules/spec_convert.md` v3.0: Added dual pipeline (batch + streaming) with applicability per direction.
- `CLAUDE.md`: Updated architecture, data flow, converters, visualizers, and gotchas for streaming.
- `README.md`: Updated test count.

## [1.1.1] - 2026-06-07

### Fixed

- **RLE converter test compatibility**: Added `coco_mask = None` placeholder in `except ImportError` block to fix test mocking when pycocotools is not installed.

### Documentation

- **SDD development workflow**: Enhanced `specs/SDD_GUIDE.md` with living-document philosophy, explicit planning step before implementation, and structured P0–P2 document sync checklist.
- **CLAUDE.md alignment**: Updated "Specs vs CLAUDE.md" to reflect the living-document approach — specs should be proactively updated when insufficient, not only when requirements change.
- **SDD_GUIDE.md bug fixes**: Fixed stale Known Gotchas count (9→13), removed duplicate formats/ directory in file listing, corrected coordinate system description to match actual native-format storage architecture.

## [1.1.0] - 2026-06-06

### Added

- **Evaluation module** (`dataflow/evaluate/`): COCO-standard evaluation for object detection and instance segmentation. Supports DetectionEvaluator (bbox IoU), SegmentationEvaluator (mask IoU), compute_pr_f1() for single-threshold quick evaluation, and full 12-metric COCO output (AP/AP50/AP75/mAP/AR with scale stratification). Verbose mode provides per-class breakdown tables.
- **YOLO prediction mode** (`--prediction` flag): `yolo2coco` now supports converting YOLO model output files (6 tokens for detection, even tokens for segmentation) with confidence scores preserved as COCO `score` field. Enables end-to-end YOLO model evaluation pipeline.
- **CLI evaluate commands**: `dataflow-cv evaluate detection` and `dataflow-cv evaluate segmentation` with `--verbose`, `--prf1`, `--prf1-iou`, `--prf1-conf`, and `--output` options.

### Documentation

- **README evaluation section**: Expanded with data preparation guide, detection vs segmentation format requirements table, numbered step-by-step workflow, and end-to-end YOLO→COCO→evaluation pipeline example.
- **README visual polish**: Added emoji section headers, mermaid pipeline diagram, centered badges, collapsible coverage table, and blockquote callouts.

## [1.0.1] - 2026-06-04

### Fixed

- **Text label positioning**: Fixed text baseline offset and background rectangle miscalculation in `_draw_text()`. Text was shifted up by `baseline` pixels and background was `2*baseline` pixels too tall.
- **Duplicate class labels for COCO**: Fixed class name being drawn twice for COCO annotations that have both bbox and segmentation. Label is now drawn once in `_draw_render_annotation()`.
- **YOLO bbox precision**: Deferred `int()` truncation to final coordinates in `YOLOVisualizer`, eliminating ±2 px unnecessary precision loss from intermediate truncation.
- **Polygon vertex guard**: Raised minimum vertex check from 2 to 3 in `_draw_polygon()` to match OpenCV requirements.
- **ColorManager algorithm**: Replaced degenerate HSV step algorithm with golden ratio conjugate hue spacing. Saturation increased from 39-41% to 78-100%, eliminating washed-out colors. All 1000 predefined colors are now unique (was 754).

### Changed

- **Format ordering convention**: All cross-format listings (docs, enums, imports, CLI help, tables, `__all__`) now follow `YOLO → LabelMe → COCO` order — simple to complex progression.
- **CLAUDE.md**: Added Format Ordering Convention section documenting the ordering rule, rationale, and scope.

## [1.0.0] - 2026-06-01

### ⚠️ Breaking Changes

- **Native-coordinate architecture**: Complete redesign of the internal data model. Handlers now store coordinates in each format's native representation instead of a unified normalized [0,1] model.
  - `DatasetAnnotations` now has a required `format` field to interpret coordinate semantics
  - `BoundingBox` and `Segmentation` no longer have `xyxy()`, `xywh_abs()`, `points_abs()` methods — converters own all coordinate transforms
  - `OriginalData` and `OriginalDataManager` removed entirely
  - `_validate_bbox()` and `_validate_segmentation_points()` are now format-aware

### Added

- **Explicit coordinate transforms in converters**: `YoloAndCocoConverter`, `LabelMeAndYoloConverter`, and `CocoAndLabelMeConverter` now implement actual coordinate transformation logic in `convert_annotations()` instead of identity pass-through
- **RenderData unified rendering pipeline**: Visualizers convert all annotations to absolute-pixel `RenderAnnotation` during `load_annotations()`, removing coordinate math from draw methods
- **Format-aware validation**: `_validate_absolute_coordinate()` for COCO/LabelMe, format dispatch in `_validate_bbox()` and `_validate_segmentation_points()`

### Changed

- **RLE converter**: `polygon_to_rle()` now accepts absolute pixel points; `rle_to_polygon()` returns absolute pixel points (no more implicit normalization)
- **COCO handler**: Stores coordinates in native COCO format (absolute pixels, top-left) — no longer normalizes to [0,1] center-based
- **LabelMe handler**: Stores coordinates in native LabelMe format (absolute pixels) — no longer normalizes
- **YOLO handler**: Simplified — removed OriginalData storage and write-path branching (YOLO coordinates were already native normalized)

### Removed

- `OriginalData` dataclass and `OriginalDataManager` static class
- `verify_lossless_roundtrip()` and all associated comparison functions
- `BoundingBox.xyxy()`, `xywh_abs()`, `has_original_data()`
- `Segmentation.points_abs()`, `has_original_data()`
- Lossless round-trip demo (`samples/label/lossless_demo.py`) and tests (`tests/label/test_lossless.py`)

### Documentation

- Complete rewrite of `specs/modules/spec_label.md`, `specs/formats/spec_conversion.md`
- Update of `specs/modules/spec_convert.md`, `specs/modules/spec_visualize.md`
- Update of format specs (`spec_yolo_format.md`, `spec_coco_format.md`, `spec_labelme_format.md`)
- Update `CLAUDE.md` and `README.md` for new architecture

## [0.6.2] - 2026-04-20

### Fixed
- **Verbose log file path handling**: Ensure consistent verbose log file path handling across visualization and conversion operations

### Documentation
- **Verbose log documentation**: Document verbose log file path handling in CLAUDE.md and README.md
- **Missing documentation details**: Add missing documentation details to CLAUDE.md and README.md

## [0.6.1] - 2026-03-31

### Fixed
- **Missing image handling**: Gracefully handle missing images in visualization with warnings instead of crashes
- **Error message completeness**: Ensure complete error messages are displayed in non-verbose mode for better debugging

### Changed
- **Visualization exit behavior**: Optimized q/ESC key exit behavior for smoother user experience

### Documentation
- **Format documentation**: Added YOLO, COCO, and LabelMe format documentation in `docs/formats/` directory
- **Chinese to English translation**: Translated Chinese comments and docstrings to English in samples and tests for better accessibility
- **Specification reorganization**: Reorganized specification files with numbered ordering for better navigation
- **CLI examples**: Updated documentation with accurate CLI examples and git commit guidelines
- **Recent feature updates**: Updated CLAUDE.md and README.md with recent features and improvements

## [0.6.0] - 2026-03-30

### ⚠️ Breaking Changes
- **CLI parameter restructuring**: Changed required parameters from options to positional arguments for `convert` and `visualize` subcommands, improving usability and consistency.
- **CLI option simplification**: Removed duplicate and redundant options, streamlined help output formatting.
- **Global options migration**: Moved remaining global options to subcommand level for better modularity.

### Added
- **RLE mask visualization**: Added support for visualizing RLE masks with semi-transparent fills in COCO format visualizations.
- **Visualization window display**: Enabled visualization window display by default with proper exception handling for headless environments.
- **Category utilities**: Added category management utilities for label handlers to support class ID mapping.
- **Logging enhancements**: Added missing `VerboseLoggingOperations` export and improved verbose logging configuration in demo scripts.
- **Specification documentation**: Added comprehensive specification documents for CLI, conversion, visualization, and logging modules.

### Changed
- **Build system migration**: Migrated from `setup.py` to modern `pyproject.toml` configuration for better packaging compatibility.
- **CLI architecture**: Further modularized CLI structure, moving global options to subcommands and simplifying parameter structures.
- **Color management**: Improved color assignment for different class IDs to ensure unique colors in visualizations.
- **Path handling**: Resolved path duplication issues in YOLO and LabelMe handlers for more reliable file operations.
- **Test and sample updates**: Updated tests and samples to match latest CLI implementation and parameter structures.

### Fixed
- **Color cycling**: Fixed `ColorManager` color cycling to match specification, ensuring consistent color assignment across visualizations.
- **Result key inconsistency**: Fixed result key inconsistency between YOLO and LabelMe handlers.
- **Decorator chain**: Corrected decorator chain for common options in CLI commands.
- **Verbose logging**: Fixed verbose logging configuration in visualization demo scripts to properly generate log files.
- **Duplicate version**: Removed duplicate `__version__` definition in CLI module.

### Documentation
- **Format documentation**: Added detailed YOLO, COCO, and LabelMe format documentation in `docs/formats/` directory.
- **Chinese to English translation**: Translated Chinese comments and docstrings to English in samples and tests for better accessibility.
- **Specification reorganization**: Reorganized specification files with numbered ordering for better navigation.
- **CLI examples**: Updated documentation with accurate CLI examples and git commit guidelines.
- **Development guidance**: Updated CLAUDE.md and README.md with comprehensive developer guidance and actual LLM information.
- **RLE documentation**: Enhanced RLE mask documentation with format comparison and pycocotools examples.

## [0.5.0] - 2026-03-15

### ⚠️ Breaking Changes
- **CLI option removal**: Removed global `--verbose` and `--overwrite` options. Verbose mode is now available as local `-v/--verbose` option for each subcommand (`convert` and `visualize`).

### Added
- **Local verbose option**: Added `-v/--verbose` option to `convert` subcommands for detailed logging during conversion operations.
- **Local verbose option**: Added `-v/--verbose` option to `visualize` subcommands for detailed logging during visualization operations.
- **Test coverage**: Added comprehensive test coverage for conversion and visualization modules.
- **Documentation**: Updated documentation with latest features and test counts.

### Changed
- **CLI architecture**: Further modularized CLI structure, moving verbose option from global to subcommand level.
- **Log output**: Optimized log output for visualization commands to reduce noise.

### Fixed
- **Empty output directories**: Improved handling of empty output directory paths in conversion operations.

### Documentation
- Added platform compatibility badges to README.
- Updated documentation with recent enhancements and test counts.

## [0.4.1] - 2026-03-14

### Added
- **Cross-platform compatibility**: Full Windows, Linux, macOS support with no platform-specific code
- **Enhanced documentation**: Updated CLI and Python API examples with segmentation support flags
- **Simplified development**: Consolidated dependency files and version constraints
- **Command alias**: Added `dataflow-cv` command alias for the existing `dataflow` command

### Changed
- **Dependency versions**: Lowered minimum versions for better compatibility with older environments
- **CLI architecture**: Modularized CLI structure for better maintainability
- **Platform compatibility**: Enhanced cross-platform support and removed all Unix-specific paths

### Fixed
- **Windows compatibility**: Fixed platform-specific code and path handling for Windows
- **Documentation**: Updated README.md and CLAUDE.md with latest features and cross-platform guidelines

### Documentation
- Added cross-platform development principles to documentation
- Updated examples with `--segmentation` flag usage for all visualization commands
- Enhanced CLAUDE.md with recent improvements and development guidelines

## [0.4.0] - 2026-03-13

### ⚠️ Breaking Changes
- **API change**: `coco_to_yolo()` function signature changed from `(coco_json_path, classes_path, output_dir)` to `(coco_json_path, output_dir)` (auto-generates class.names)
- **API change**: `labelme_to_yolo()` function signature changed from `(label_dir, output_dir)` to `(label_dir, classes_path, output_dir)`
- Unified YOLO conversion behavior across COCO and LabelMe converters

### Added
- YOLO visualizer debug logging for class extraction and color assignment
- Enhanced color distinction for many classes in visualization
- CLI option `-v` as shorthand for `--version` when used alone
- Documentation updates: architecture clarification and development guidelines

### Changed
- Unified COCO and LabelMe to YOLO conversion behavior
- Updated YOLO detection annotation to LabelMe rectangle conversion
- Improved test parameter calls and directory expectations for LabelMe to YOLO conversion

### Fixed
- YOLO detection annotations correctly convert to LabelMe rectangles
- Test parameter calls and directory expectations in LabelMe to YOLO tests

### Documentation
- Updated CLAUDE.md with `-v` dual-purpose note
- Added "Development" section to README.md linking to CLAUDE.md
- Updated CLAUDE.md with architecture details about label handlers and unified format

## [0.3.1] - 2026-03-11

### Added
- Added PyPI version badge and GitHub Actions build status badge to README.md
- Added GitHub Actions workflow for PyPI publishing (python-publish.yml)
- Added detailed YOLO, LabelMe, and COCO format documentation in docs/ directory

### Changed
- Updated GitHub workflow name from "Upload Python Package" to "Publish"
- Updated version from 0.3.0 to 0.3.1

### Fixed
- Fixed color assignment in LabelMe visualization to assign distinct colors per class (previously all annotations used same color)

### Documentation
- Added Documentation section to table of contents in README.md
- Updated README.md and CLAUDE.md with latest project structure and segmentation support
- Added generic.py to visualize module documentation
- Added all converter test files to project structure

## [0.3.0] - 2026-03-11

### Added
- Complete LabelMe format conversion support with bidirectional conversion between LabelMe, COCO, and YOLO formats
- Added `coco2labelme`, `labelme2coco`, `labelme2yolo`, and `yolo2labelme` conversion commands
- Added LabelMe visualization module (`dataflow visualize labelme`) and Python API function `visualize_labelme()`
- Added comprehensive LabelMe conversion examples in `samples/` directory

### Changed
- Refactored conversion module structure for better maintainability and extensibility
- Updated module organization following task-sub-task pattern

### Fixed
- Fixed image path resolution in LabelMeVisualizer

### Documentation
- Updated README Installation section with detailed instructions
- Added LabelMe conversion examples and documentation

## [0.2.1] - 2026-03-10

### Changed
- Updated version from 0.2.0 to 0.2.1

## [0.2.0] - 2026-03-08

### Added
- Added YOLO and COCO visualization module with CLI support (`dataflow visualize yolo` and `dataflow visualize coco`)
- Added Python API convenience functions: `visualize_yolo()` and `visualize_coco()`
- Added comprehensive visualization examples in `samples/` directory

### Changed
- Refactored COCO-YOLO conversion to be batch-first with new interfaces
- Improved command-line interface options and error handling
- Cleaned up and consolidated codebase structure
- Updated version from 0.1.1 to 0.2.0

### Fixed
- Fixed CLI interface options and improved user experience

### Documentation
- Updated CLAUDE.md with detailed usage and architecture guidelines
- Added DeepSeek-V3.2 AI model information to git commit guidelines
- Updated git commit guidelines with proper AI model attribution

## [0.1.1] - 2026-03-07

### Added
- Added `setup.py` to enable editable installations via `python setup.py develop`
- Added CHANGELOG.md file

### Changed
- Updated version from 0.1.0 to 0.1.1
- Updated dependency versions:
  - `numpy>=2.0.0` (from >=1.19.0)
  - `opencv-python>=4.8.0` (from >=4.5.0)
  - `Pillow>=10.0.0` (from >=8.0.0)
  - `click>=8.1.0` (from >=8.0.0)
- Updated installation documentation in README.md and CLAUDE.md:
  - Added `python setup.py develop` for editable installation
  - Changed `pip install -e .` to `pip install .` for regular installation
- Updated `pyproject.toml`:
  - Fixed license format to use SPDX expression
  - Removed deprecated license classifier
  - Limited setuptools version to <70 for compatibility

### Fixed
- Fixed build warnings related to license format and classifiers
- Fixed license configuration in pyproject.toml (use `{text = "MIT"}` format)
- Added custom develop command to enable `python setup.py develop` for editable installation
- Note: `pip install -e .` may not work due to setuptools compatibility issues; use `python setup.py develop` instead
  - With editable installation, use `python -m dataflow.cli` instead of `dataflow` command

## [0.1.0] - 2026-03-06

### Added
- Initial release with format conversion between LabelMe, COCO, and YOLO formats
- Batch conversion support for all conversion directions
- Single-image and batch visualization
- Command-line interface with `convert` and `visualize` subcommands
- Python API for programmatic access