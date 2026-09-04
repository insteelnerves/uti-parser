"""
Converts VLM/LLM text outputs containing classification predictions into
standardized NovaVision classification prediction format.

Class mapping behavior:
- If ConfigClasses is provided:
    Known classes get their index from the config list.
    Unknown classes get class_id -1.
- If ConfigClasses is empty:
    Classes are auto-detected from parsed model output.
    Class IDs are generated deterministically.
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.Parser.src.utils.response import build_vlm_as_classifier_response
from components.Parser.src.models.PackageModel import PackageModel
from components.Parser.src.utils.parser_helper import ParserHelper

class VLMAsClassifier(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)

        self.request.model = PackageModel(**(self.request.data))

        self.input_image = self.request.get_param("inputImage")
        self.raw_text = self.request.get_param("inputRawText")
        self.model_type = self.request.get_param("ConfigClassifierModelType") or "auto"
        self.classes_raw = self.request.get_param("ConfigClasses")

        self.error_status = False
        self.inference_id = ParserHelper.new_inference_id()
        self.output_data = {}

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
            return build_vlm_as_classifier_response(context=self)

        height, width = img.value.shape[:2]

        parsed_data = ParserHelper.parse_json_text(self.raw_text)

        if parsed_data is None:
            self.error_status = True
            return build_vlm_as_classifier_response(context=self)

        classification_format = ParserHelper.detect_classification_format(parsed_data)

        if classification_format == "unknown":
            self.error_status = True
            return build_vlm_as_classifier_response(context=self)

        classification_items = ParserHelper.extract_classification_items(parsed_data)

        class_list = ParserHelper.parse_classes(self.classes_raw)
        class_map = ParserHelper.build_class_map(class_list)
        normalized_class_map = ParserHelper.build_normalized_class_map(class_list)

        predictions = []

        for item in classification_items:
            class_name = str(item.get("class_name", ""))

            if not class_name:
                continue

            confidence = ParserHelper.clamp(item.get("confidence", 0.0))

            resolved_class_name, class_id = ParserHelper.resolve_class_id(
                class_name=class_name,
                class_list=class_list,
                class_map=class_map,
                normalized_class_map=normalized_class_map,
                auto_when_empty=True
            )

            if not resolved_class_name:
                continue

            predictions.append({
                "class_name": resolved_class_name,
                "class_id": int(class_id),
                "confidence": float(confidence)
            })

        if classification_format == "single":
            if not predictions:
                self.error_status = True
                return build_vlm_as_classifier_response(context=self)

            top_prediction = max(
                predictions,
                key=lambda prediction: prediction["confidence"]
            )

            self.output_data = {
                "type": "classification",
                "width": int(width),
                "height": int(height),
                "inference_id": self.inference_id,
                "parent_id": self.inference_id,
                "top": top_prediction,
                "confidence": top_prediction["confidence"],
                "predictions": [
                    top_prediction
                ]
            }

        else:
            self.output_data = {
                "type": "multi-label-classification",
                "width": int(width),
                "height": int(height),
                "inference_id": self.inference_id,
                "parent_id": self.inference_id,
                "predicted_classes": predictions,
                "predictions": {
                    "predicted_classes": predictions
                }
            }

        return build_vlm_as_classifier_response(context=self)

if "__main__" == __name__:
    Executor(sys.argv[1]).run()
