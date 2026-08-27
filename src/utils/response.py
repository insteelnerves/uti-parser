from sdks.novavision.src.helper.package import PackageHelper
from components.Parser.src.models.PackageModel import (
    PackageModel,
    PackageConfigs,
    ConfigExecutor,
    JsonParser,
    JsonParserOutputs,
    JsonParserResponse,
    OutputData,
    OutputErrorStatus,
    VLMAsDetector,
    VLMAsDetectorOutputs,
    VLMAsDetectorResponse,
    OutputDetections,
    OutputInferenceId,
    VLMAsClassifier,
    VLMAsClassifierOutputs,
    VLMAsClassifierResponse
)


def build_json_parser_response(context):
    output_data = OutputData(value=context.output_data)
    output_error_status = OutputErrorStatus(value=context.error_status)

    outputs = JsonParserOutputs(
        outputData=output_data,
        outputErrorStatus=output_error_status
    )

    response = JsonParserResponse(outputs=outputs)
    executor = JsonParser(value=response)
    config_executor = ConfigExecutor(value=executor)
    package_configs = PackageConfigs(executor=config_executor)

    package = PackageHelper(
        packageModel=PackageModel,
        packageConfigs=package_configs
    )

    return package.build_model(context)


def build_vlm_as_detector_response(context):
    output_detections = OutputDetections(value=context.output_detections)
    output_error_status = OutputErrorStatus(value=context.error_status)
    output_inference_id = OutputInferenceId(value=context.inference_id)

    outputs = VLMAsDetectorOutputs(
        outputDetections=output_detections,
        outputErrorStatus=output_error_status,
        outputInferenceId=output_inference_id
    )

    response = VLMAsDetectorResponse(outputs=outputs)
    executor = VLMAsDetector(value=response)
    config_executor = ConfigExecutor(value=executor)
    package_configs = PackageConfigs(executor=config_executor)

    package = PackageHelper(
        packageModel=PackageModel,
        packageConfigs=package_configs
    )

    return package.build_model(context)


def build_vlm_as_classifier_response(context):
    output_data = OutputData(value=context.output_data)
    output_error_status = OutputErrorStatus(value=context.error_status)
    output_inference_id = OutputInferenceId(value=context.inference_id)

    outputs = VLMAsClassifierOutputs(
        outputData=output_data,
        outputErrorStatus=output_error_status,
        outputInferenceId=output_inference_id
    )

    response = VLMAsClassifierResponse(outputs=outputs)
    executor = VLMAsClassifier(value=response)
    config_executor = ConfigExecutor(value=executor)
    package_configs = PackageConfigs(executor=config_executor)

    package = PackageHelper(
        packageModel=PackageModel,
        packageConfigs=package_configs
    )

    return package.build_model(context)
