"""
Concrete evaluator implementations for DataFlow-CV.

Provides DetectionEvaluator (bbox IoU) and SegmentationEvaluator (mask IoU).
"""

from typing import Any, Optional

from .base import BaseEvaluator


class DetectionEvaluator(BaseEvaluator):
    """Object detection evaluation using bounding box IoU.

    Evaluates predicted bounding boxes against ground truth using the
    COCO detection protocol (``iouType='bbox'``).  GT and DT must be in
    COCO format; DT annotations must include a ``score`` field.

    Example usage::

        evaluator = DetectionEvaluator(
            log_config=LogConfig(name="eval", verbose=True)
        )
        result = evaluator.evaluate("gt.json", "dt.json")
        print(result.get_summary())
    """

    def __init__(self, log_config=None):
        super().__init__(log_config=log_config)

    def _iou_type(self) -> str:
        return "bbox"

    def _create_cocoeval(self, coco_gt: Any, coco_dt: Any) -> Any:
        """Create COCOeval for detection (bbox IoU).

        ``_validate_coco_available()`` is called by
        :meth:`BaseEvaluator.evaluate` before this method, so no
        redundant check is needed here.
        """
        from pycocotools.cocoeval import COCOeval

        return COCOeval(coco_gt, coco_dt, iouType="bbox")


class SegmentationEvaluator(BaseEvaluator):
    """Instance segmentation evaluation using mask IoU.

    Evaluates predicted segmentation masks against ground truth using the
    COCO segmentation protocol (``iouType='segm'``).  GT and DT must
    include ``segmentation`` data (polygon or RLE format).

    Requires pycocotools.

    Example usage::

        evaluator = SegmentationEvaluator(
            log_config=LogConfig(name="eval", verbose=True)
        )
        result = evaluator.evaluate("gt_segm.json", "dt_segm.json")
        print(result.get_summary())
    """

    def __init__(self, log_config=None):
        super().__init__(log_config=log_config)

    def _iou_type(self) -> str:
        return "segm"

    def _create_cocoeval(self, coco_gt: Any, coco_dt: Any) -> Any:
        """Create COCOeval for segmentation (mask IoU).

        ``_validate_coco_available()`` is called by
        :meth:`BaseEvaluator.evaluate` before this method, so no
        redundant check is needed here.
        """
        from pycocotools.cocoeval import COCOeval

        return COCOeval(coco_gt, coco_dt, iouType="segm")
