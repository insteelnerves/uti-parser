"""
Converts VLM/LLM text outputs containing object detection predictions into
standardized NovaVision object detection format.

Class mapping behavior:
- If ConfigClasses is provided:
    Known classes get their index from the config list.
    Unknown classes get class_id -1.
- If ConfigClasses is empty:
    Classes are auto-detected from parsed model output.
    Class IDs are generated deterministically.

Coordinate format behavior:
- auto:
    Parser tries to detect normalized 0-1, normalized 0-1000, or pixel coordinates.
- normalized-0-1:
    Coordinates are scaled from [0, 1] to image dimensions.
- normalized-0-1000:
    Coordinates are scaled from [0, 1000] to image dimensions.
- pixel:
    Coordinates are kept as pixel values.
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.base.model import Detection, BoundingBox
from sdks.novavision.src.helper.executor import Executor
from components.Parser.src.utils.response import build_vlm_as_detector_response
from components.Parser.src.models.PackageModel import PackageModel
from components.Parser.src.utils.parser_helper import ParserHelper


class VLMAsDetector(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)

        self.request.model = PackageModel(**(self.request.data))

        self.input_image = self.request.get_param("inputImage")
        self.raw_text = self.request.get_param("inputRawText")

        self.model_type = self.request.get_param("ConfigModelType") or "auto"
        self.task_type = self.request.get_param("ConfigTaskType") or "object-detection"
        self.coordinate_format = self.request.get_param("ConfigCoordinateFormat") or "auto"
        self.classes_raw = self.request.get_param("ConfigClasses")

        self.error_status = False
        self.inference_id = ParserHelper.new_inference_id()
        self.output_detections = []

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def run(self):
        img = Image.get_frame(
            img=self.input_image,
            redis_db=self.redis_db
        )

        if img is None or img.value is None:
            self.error_status = True
            return build_vlm_as_detector_response(context=self)

        height, width = img.value.shape[:2]

        parsed_data = ParserHelper.parse_json_text(self.raw_text)

        if parsed_data is None:
            self.error_status = True
            return build_vlm_as_detector_response(context=self)

        detection_items = ParserHelper.extract_detection_items(parsed_data)

        if detection_items is None:
            self.error_status = True
            return build_vlm_as_detector_response(context=self)

        class_list = ParserHelper.parse_classes(self.classes_raw)
        class_map = ParserHelper.build_class_map(class_list)
        normalized_class_map = ParserHelper.build_normalized_class_map(class_list)

        detections = []

        for item in detection_items:
            if not isinstance(item, dict):
                continue

            box_result = ParserHelper.extract_box_with_source(item)

            if box_result is None:
                continue

            box, box_source = box_result

            normalized_box = ParserHelper.normalize_box(
                box=box,
                width=width,
                height=height,
                source_key=box_source,
                coordinate_format=self.coordinate_format,
                model_type=self.model_type
            )

            if normalized_box is None:
                continue

            class_label = ParserHelper.get_string(
                item,
                ParserHelper.CLASS_KEYS,
                default=None
            )

            if self.task_type == "region-proposal":
                class_label = "roi"

            if class_label is None:
                class_label = ""

            default_confidence = 1.0 if self.model_type == "florence-2" else 0.0

            confidence = ParserHelper.parse_confidence(
                item,
                default=default_confidence
            )

            auto_when_empty = True

            if self.model_type == "florence-2" and self.task_type in [
                "region-proposal",
                "open-vocabulary-object-detection"
            ]:
                auto_when_empty = False

            resolved_class_label, class_id = ParserHelper.resolve_class_id(
                class_name=class_label,
                class_list=class_list,
                class_map=class_map,
                normalized_class_map=normalized_class_map,
                auto_when_empty=auto_when_empty
            )

            if self.model_type == "florence-2" and self.task_type == "region-proposal" and class_id == -1:
                class_id = 0

            left, top, box_width, box_height = normalized_box

            detection = Detection(
                boundingBox=BoundingBox(
                    left=float(left),
                    top=float(top),
                    width=float(box_width),
                    height=float(box_height)
                ),
                confidence=float(confidence),
                classLabel=str(resolved_class_label or class_label),
                classId=int(class_id)
            )

            detections.append(detection)

        self.output_detections = detections

        return build_vlm_as_detector_response(context=self)


if "__main__" == __name__:
    Executor(sys.argv[1]).run()
