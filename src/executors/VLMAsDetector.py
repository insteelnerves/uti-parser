"""
Converts VLM/LLM text outputs containing object detection predictions into
standardized NovaVision object detection format.
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

        self.model_type = self.request.get_param("ConfigModelType")
        self.task_type = self.request.get_param("ConfigTaskType")
        self.classes_raw = self.request.get_param("ConfigClasses")

        self.error_status = False
        self.inference_id = ParserHelper.new_inference_id()
        self.output_detections = []

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def _get_class_id(self, class_label: str, class_map: dict, class_list: list) -> int:
        if class_label in class_map:
            return class_map[class_label]

        if self.model_type == "florence-2":
            if self.task_type == "region-proposal":
                return class_map.get("roi", 0)

            if self.task_type == "open-vocabulary-object-detection":
                return -1

            return ParserHelper.generate_class_id(class_label)

        return -1

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

        detections = []

        for item in detection_items:
            if not isinstance(item, dict):
                continue

            box = ParserHelper.extract_box(item)
            normalized_box = ParserHelper.normalize_box(
                box=box,
                width=width,
                height=height
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

            class_id = self._get_class_id(
                class_label=class_label,
                class_map=class_map,
                class_list=class_list
            )

            left, top, box_width, box_height = normalized_box

            detection = Detection(
                boundingBox=BoundingBox(
                    left=float(left),
                    top=float(top),
                    width=float(box_width),
                    height=float(box_height)
                ),
                confidence=float(confidence),
                classLabel=str(class_label),
                classId=int(class_id)
            )

            detections.append(detection)

        self.output_detections = detections

        return build_vlm_as_detector_response(context=self)


if "__main__" == __name__:
    Executor(sys.argv[1]).run()
