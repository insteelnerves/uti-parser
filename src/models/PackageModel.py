from pydantic import Field, validator
from typing import List, Optional, Union, Literal
from sdks.novavision.src.base.model import (
    Package,
    Image,
    Detection,
    Inputs,
    Configs,
    Outputs,
    Response,
    Request,
    Output,
    Input,
    Config
)

class InputImage(Input):
    name: Literal["inputImage"] = "inputImage"
    value: Union[List[Image], Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get("value")
        if isinstance(value, Image):
            return "object"
        elif isinstance(value, list):
            return "list"

    class Config:
        title = "Image"


class InputRawText(Input):
    """
    Raw VLM/LLM output.
    Can be:
    - Plain text or JSON string
    - Markdown-wrapped JSON
    - Already parsed dict or list (from upstream packages like GCP Vision)
    """
    name: Literal["inputRawText"] = "inputRawText"
    value: Union[str, dict, list]
    type: str = "string"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get("value")
        if isinstance(value, str):
            return "string"
        elif isinstance(value, list):
            return "list"
        elif isinstance(value, dict):
            return "object"

    class Config:
        title = "Raw Text / Data"


class OutputData(Output):
    name: Literal["outputData"] = "outputData"
    value: Union[dict, list]
    type: str = "object"

    class Config:
        title = "Output Data"


class OutputErrorStatus(Output):
    name: Literal["outputErrorStatus"] = "outputErrorStatus"
    value: bool
    type: Literal["bool"] = "bool"

    class Config:
        title = "Error Status"


class OutputInferenceId(Output):
    name: Literal["outputInferenceId"] = "outputInferenceId"
    value: str
    type: Literal["string"] = "string"

    class Config:
        title = "Inference ID"


class OutputDetections(Output):
    name: Literal["outputDetections"] = "outputDetections"
    value: List[Detection]
    type: Literal["list"] = "list"

    class Config:
        title = "Detections"


class ConfigExpectedFields(Config):
    """
    Expected JSON fields to extract.
    Supports root keys and nested paths.

    Examples:
    name,age,result
    employees[0].firstName
    employees[*].firstName
    employees.firstName
    """
    name: Literal["ConfigExpectedFields"] = "ConfigExpectedFields"
    value: str = Field(default="")
    type: Literal["string"] = "string"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["field1,field2,field3"] = "field1,field2,field3"

    class Config:
        title = "Expected Fields"
        json_schema_extra = {
            "shortDescription": "Expected Fields"
        }

class OptionModelAuto(Config):
    name: Literal["optionAuto"] = "optionAuto"
    value: Literal["auto"] = "auto"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Auto"

class OptionModelGCPVision(Config):
    name: Literal["optionGCPVision"] = "optionGCPVision"
    value: Literal["gcp-vision"] = "gcp-vision"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "GCP Vision"

class OptionModelQwenAI(Config):
    name: Literal["optionQwenAI"] = "optionQwenAI"
    value: Literal["qwen-ai"] = "qwen-ai"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Qwen AI"

class OptionModelKimiAI(Config):
    name: Literal["optionKimiAI"] = "optionKimiAI"
    value: Literal["kimi-ai"] = "kimi-ai"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Kimi AI"

class OptionModelOpenAI(Config):
    name: Literal["optionOpenAI"] = "optionOpenAI"
    value: Literal["openai"] = "openai"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "OpenAI"

class OptionModelGoogleGemini(Config):
    name: Literal["optionGoogleGemini"] = "optionGoogleGemini"
    value: Literal["google-gemini"] = "google-gemini"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Google Gemini"

class OptionModelAnthropicClaude(Config):
    name: Literal["optionAnthropicClaude"] = "optionAnthropicClaude"
    value: Literal["anthropic-claude"] = "anthropic-claude"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Anthropic Claude"


class OptionModelSpaceXAI(Config):
    name: Literal["optionSpaceXAI"] = "optionSpaceXAI"
    value: Literal["spacexai"] = "spacexai"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "SpaceXAI"


class OptionModelFlorence2(Config):
    name: Literal["optionFlorence2"] = "optionFlorence2"
    value: Literal["florence-2"] = "florence-2"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Florence-2"

class OptionCoordinateAuto(Config):
    name: Literal["optionCoordinateAuto"] = "optionCoordinateAuto"
    value: Literal["auto"] = "auto"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Auto"


class OptionCoordinateNormalized01(Config):
    name: Literal["optionCoordinateNormalized01"] = "optionCoordinateNormalized01"
    value: Literal["normalized-0-1"] = "normalized-0-1"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Normalized 0-1"


class OptionCoordinateNormalized01000(Config):
    name: Literal["optionCoordinateNormalized01000"] = "optionCoordinateNormalized01000"
    value: Literal["normalized-0-1000"] = "normalized-0-1000"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Normalized 0-1000"


class OptionCoordinatePixel(Config):
    name: Literal["optionCoordinatePixel"] = "optionCoordinatePixel"
    value: Literal["pixel"] = "pixel"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Pixel"

class ConfigCoordinateFormat(Config):
    """
    Select the coordinate format used in the detection output.

    Auto:
    - 0-1 normalized coordinates are scaled to image size.
    - 0-1000 normalized coordinates are scaled to image size.
    - Pixel coordinates are kept as-is.

    Normalized 0-1:
    - Coordinates are expected in [0, 1] range.

    Normalized 0-1000:
    - Coordinates are expected in [0, 1000] range.

    Pixel:
    - Coordinates are expected as absolute pixel values.
    """
    name: Literal["ConfigCoordinateFormat"] = "ConfigCoordinateFormat"
    value: Union[
        OptionCoordinateAuto,
        OptionCoordinateNormalized01,
        OptionCoordinateNormalized01000,
        OptionCoordinatePixel
    ] = Field(default_factory=OptionCoordinateAuto)
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Coordinate Format"
        json_schema_extra = {
            "shortDescription": "Coordinate Format"
        }

class ConfigModelType(Config):
    """
    Select the VLM/LLM model type producing the detection JSON.
    Auto mode uses a generic and safe parsing behavior.
    """
    name: Literal["ConfigModelType"] = "ConfigModelType"
    value: Union[
        OptionModelAuto,
        OptionModelOpenAI,
        OptionModelGoogleGemini,
        OptionModelAnthropicClaude,
        OptionModelSpaceXAI,
        OptionModelFlorence2,
        OptionModelGCPVision,
        OptionModelQwenAI,
        OptionModelKimiAI
    ] = Field(default_factory=OptionModelAuto)
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Model Type"
        json_schema_extra = {
            "shortDescription": "Model Type"
        }

class ConfigClassifierModelType(Config):
    """
    Select the VLM/LLM model type producing the classification output.
    Auto mode uses a generic and safe parsing behavior.
    """
    name: Literal["ConfigClassifierModelType"] = "ConfigClassifierModelType"
    value: Union[
        OptionModelAuto,
        OptionModelOpenAI,
        OptionModelGoogleGemini,
        OptionModelAnthropicClaude,
        OptionModelSpaceXAI,
        OptionModelGCPVision,
        OptionModelQwenAI,
        OptionModelKimiAI
    ] = Field(default_factory=OptionModelAuto)
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Model Type"
        json_schema_extra = {
            "shortDescription": "Model Type"
        }

class OptionTaskObjectDetection(Config):
    name: Literal["optionObjectDetection"] = "optionObjectDetection"
    value: Literal["object-detection"] = "object-detection"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Object Detection"


class OptionTaskOpenVocabularyObjectDetection(Config):
    name: Literal["optionOpenVocabularyObjectDetection"] = "optionOpenVocabularyObjectDetection"
    value: Literal["open-vocabulary-object-detection"] = "open-vocabulary-object-detection"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Open Vocabulary Object Detection"

class OptionTaskObjectDetectionAndCaption(Config):
    name: Literal["optionObjectDetectionAndCaption"] = "optionObjectDetectionAndCaption"
    value: Literal["object-detection-and-caption"] = "object-detection-and-caption"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Object Detection and Caption"


class OptionTaskPhraseGroundedObjectDetection(Config):
    name: Literal["optionPhraseGroundedObjectDetection"] = "optionPhraseGroundedObjectDetection"
    value: Literal["phrase-grounded-object-detection"] = "phrase-grounded-object-detection"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Phrase Grounded Object Detection"


class OptionTaskRegionProposal(Config):
    name: Literal["optionRegionProposal"] = "optionRegionProposal"
    value: Literal["region-proposal"] = "region-proposal"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Region Proposal"


class OptionTaskOcrWithTextDetection(Config):
    name: Literal["optionOcrWithTextDetection"] = "optionOcrWithTextDetection"
    value: Literal["ocr-with-text-detection"] = "ocr-with-text-detection"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "OCR with Text Detection"


class ConfigTaskType(Config):
    """
    Select the task type used by the VLM/LLM detection output.
    """
    name: Literal["ConfigTaskType"] = "ConfigTaskType"
    value: Union[
        OptionTaskObjectDetection,
        OptionTaskOpenVocabularyObjectDetection,
        OptionTaskObjectDetectionAndCaption,
        OptionTaskPhraseGroundedObjectDetection,
        OptionTaskRegionProposal,
        OptionTaskOcrWithTextDetection
    ] = Field(default_factory=OptionTaskObjectDetection)
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Task Type"
        json_schema_extra = {
            "shortDescription": "Task Type"
        }


class ConfigClasses(Config):
    """
    Optional class list used for mapping class names to class IDs.

    If provided, known classes are mapped to their index in this list.
    Unknown classes receive class_id -1.

    If left empty, classes are auto-detected from the parsed model output.

    Example: person,car,dog
    """
    name: Literal["ConfigClasses"] = "ConfigClasses"
    value: str = Field(default="")
    type: Literal["string"] = "string"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["Optional: person,car,dog"] = "Optional: person,car,dog"

    class Config:
        title = "Classes (Optional)"
        json_schema_extra = {
            "shortDescription": "Known Classes (Optional)"
        }


class JsonParserInputs(Inputs):
    inputRawText: InputRawText


class JsonParserConfigs(Configs):
    expectedFields: ConfigExpectedFields


class JsonParserOutputs(Outputs):
    outputData: OutputData
    outputErrorStatus: OutputErrorStatus


class JsonParserRequest(Request):
    inputs: Optional[JsonParserInputs]
    configs: JsonParserConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class JsonParserResponse(Response):
    outputs: JsonParserOutputs


class JsonParser(Config):
    """
    Parses raw JSON or Markdown-wrapped JSON strings and extracts expected fields.
    """
    name: Literal["JsonParser"] = "JsonParser"
    value: Union[JsonParserRequest, JsonParserResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "JSON Parser"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


class VLMAsDetectorInputs(Inputs):
    inputImage: InputImage
    inputRawText: InputRawText


class VLMAsDetectorConfigs(Configs):
    modelType: ConfigModelType = Field(default_factory=ConfigModelType)
    taskType: ConfigTaskType = Field(default_factory=ConfigTaskType)
    coordinateFormat: ConfigCoordinateFormat = Field(default_factory=ConfigCoordinateFormat)
    classes: ConfigClasses = Field(default_factory=ConfigClasses)


class VLMAsDetectorOutputs(Outputs):
    outputDetections: OutputDetections
    outputErrorStatus: OutputErrorStatus
    outputInferenceId: OutputInferenceId


class VLMAsDetectorRequest(Request):
    inputs: Optional[VLMAsDetectorInputs]
    configs: VLMAsDetectorConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class VLMAsDetectorResponse(Response):
    outputs: VLMAsDetectorOutputs


class VLMAsDetector(Config):
    """
    Converts VLM/LLM text outputs containing detection results into
    standardized NovaVision detection predictions.
    """
    name: Literal["VLMAsDetector"] = "VLMAsDetector"
    value: Union[VLMAsDetectorRequest, VLMAsDetectorResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "VLM As Detector"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


class VLMAsClassifierInputs(Inputs):
    inputImage: InputImage
    inputRawText: InputRawText


class VLMAsClassifierConfigs(Configs):
    modelType: ConfigClassifierModelType = Field(default_factory=ConfigClassifierModelType)
    classes: ConfigClasses = Field(default_factory=ConfigClasses)


class VLMAsClassifierOutputs(Outputs):
    outputData: OutputData
    outputErrorStatus: OutputErrorStatus
    outputInferenceId: OutputInferenceId


class VLMAsClassifierRequest(Request):
    inputs: Optional[VLMAsClassifierInputs]
    configs: VLMAsClassifierConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class VLMAsClassifierResponse(Response):
    outputs: VLMAsClassifierOutputs


class VLMAsClassifier(Config):
    """
    Converts VLM/LLM text outputs containing classification results into
    standardized NovaVision classification predictions.
    """
    name: Literal["VLMAsClassifier"] = "VLMAsClassifier"
    value: Union[VLMAsClassifierRequest, VLMAsClassifierResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "VLM As Classifier"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


class ConfigExecutor(Config):
    """
    Select which parsing operation to perform.
    """
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[JsonParser, VLMAsDetector, VLMAsClassifier]
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Task"
        json_schema_extra = {
            "shortDescription": "Select Task"
        }


class PackageConfigs(Configs):
    executor: ConfigExecutor


class PackageModel(Package):
    configs: PackageConfigs
    type: Literal["component"] = "component"
    name: Literal["Parser"] = "Parser"
