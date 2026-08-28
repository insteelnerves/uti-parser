"""
Parses raw JSON or Markdown-wrapped JSON strings and extracts expected fields.
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.Parser.src.utils.response import build_json_parser_response
from components.Parser.src.models.PackageModel import PackageModel
from components.Parser.src.utils.parser_helper import ParserHelper

class JsonParser(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)

        self.request.model = PackageModel(**(self.request.data))

        self.raw_text = self.request.get_param("inputRawText")
        self.expected_fields_raw = self.request.get_param("ConfigExpectedFields")

        self.error_status = False
        self.output_data = {}

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def run(self):
        expected_fields = ParserHelper.parse_expected_fields(self.expected_fields_raw)
        parsed_data = ParserHelper.parse_json_text(self.raw_text)

        if not expected_fields:
            if parsed_data is None:
                self.error_status = True
                self.output_data = {}
            elif isinstance(parsed_data, dict) or isinstance(parsed_data, list):
                self.error_status = False
                self.output_data = parsed_data
            else:
                self.error_status = False
                self.output_data = {
                    "value": parsed_data
                }

            return build_json_parser_response(context=self)

        output = {}

        for field in expected_fields:
            value, complete = ParserHelper.get_nested_value(
                data=parsed_data,
                path=field
            )

            output[field] = value

            if not complete:
                self.error_status = True

        self.output_data = output

        return build_json_parser_response(context=self)

if "__main__" == __name__:
    Executor(sys.argv[1]).run()
