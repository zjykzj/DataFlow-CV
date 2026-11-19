"""
Concrete evaluator implementations for DataFlow-CV.

Provides DetectionEvaluator (bbox IoU) and SegmentationEvaluator (mask IoU).
"""

from typing import Any

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


class KeypointEvaluator(BaseEvaluator):
    """Pose estimation evaluation using OKS (Object Keypoint Similarity).

    Evaluates predicted keypoints against ground truth using the COCO
    keypoint protocol (``iouType='keypoints'``).  GT and DT must include
    ``keypoints`` data (flat ``[x, y, v, ...]`` triplets); DT annotations
    must include a ``score`` field.

    OKS sigmas resolve per ``spec_evaluate.md`` §3.4: explicit ``sigmas``
    parameter → COCO official 17-keypoint sigmas (k == 17) → uniform
    0.05 with a WARNING.

    Example usage::

        evaluator = KeypointEvaluator(
            log_config=LogConfig(name="eval", verbose=True)
        )
        result = evaluator.evaluate("gt_pose.json", "dt_pose.json")
        print(result.get_summary())
    """

    def __init__(self, log_config=None, sigmas=None):
        super().__init__(log_config=log_config)
        self.sigmas = sigmas

    def _iou_type(self) -> str:
        return "keypoints"

    def _create_cocoeval(self, coco_gt: Any, coco_dt: Any) -> Any:
        """Create COCOeval for keypoints (OKS) with resolved sigmas.

        ``_validate_coco_available()`` is called by
        :meth:`BaseEvaluator.evaluate` before this method, so no
        redundant check is needed here.
        """
        from pycocotools.cocoeval import COCOeval

        from .utils import resolve_kpt_sigmas

        coco_eval = COCOeval(coco_gt, coco_dt, iouType="keypoints")

        k = self._detect_k(coco_gt)
        resolved, warning = resolve_kpt_sigmas(k, self.sigmas)
        if warning:
            self._log_warning(warning)
        coco_eval.params.kpt_oks_sigmas = resolved

        return coco_eval

    @staticmethod
    def _detect_k(coco_gt: Any) -> int:
        """Detect the dataset keypoint count from the first GT with keypoints."""
        for ann in coco_gt.loadAnns(coco_gt.getAnnIds()):
            kpts = ann.get("keypoints")
            if kpts:
                return len(kpts) // 3
        return 0
