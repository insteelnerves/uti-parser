import json
import re
import uuid
import zlib

from typing import Any, Dict, List, Optional, Tuple, Union


class ParserHelper:
    CLASS_KEYS = [
        "class_name",
        "className",
        "class",
        "label",
        "object",
        "object_name",
        "name",
        "category",
        "category_name",
        "description",
        "tag",
        "entity",
        "text"
    ]

    CONFIDENCE_KEYS = [
        "confidence",
        "score",
        "prob",
        "probability",
        "conf",
        "topicality",
        "likelihood",
        "scores",
        "probabilities"
    ]

    DETECTION_LIST_KEYS = [
        "detections",
        "predictions",
        "objects",
        "boxes",
        "results",
        "items",
        "data",
        "labels",
        "localizedObjectAnnotations",
        "labelAnnotations",
        "faceAnnotations",
        "textAnnotations",
        "annotations",
        "responses"
    ]

    BOX_KEYS = [
        "boundingBox",
        "bounding_box",
        "bbox",
        "box",
        "box_2d",
        "coordinates",
        "coords",
        "rect",
        "region",
        "boundingPoly",
        "normalizedVertices",
        "vertices",
        "location"
    ]

    @staticmethod
    def new_inference_id() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def _parse_json_text_raw(raw_text: Any) -> Optional[Union[dict, list]]:
        """
        Parses raw JSON text or Markdown-wrapped JSON text.

        Parse order:
        1. Strict JSON.
        2. Raw decode on original text.
        3. Repair pseudo-JSON, then strict JSON on repaired text.
        4. Salvage complete objects from truncated text.
        5. Raw decode on repaired text as a last fallback.

        Salvage runs BEFORE raw_decode on repaired text so that a truncated
        list returns ALL complete items, not only the first one.
        """
        if raw_text is None:
            return None

        if isinstance(raw_text, (dict, list)):
            return raw_text

        text = str(raw_text).strip()

        if not text:
            return None

        candidates = []

        markdown_matches = re.findall(
            r"```(?:json)?\s*([\s\S]*?)```",
            text,
            re.IGNORECASE
        )

        if markdown_matches:
            candidates.append(markdown_matches[0].strip())

        candidates.append(text)

        for candidate in candidates:
            # 1. Strict JSON.
            parsed = ParserHelper._loads(candidate)

            if parsed is not None:
                return parsed

            # 2. Raw decode on original text.
            parsed = ParserHelper._raw_decode(candidate)

            if parsed is not None:
                return parsed

            # 3. Repair pseudo-JSON.
            repaired = ParserHelper._repair_json_like_text(candidate)

            # 4. Strict JSON on repaired text.
            parsed = ParserHelper._loads(repaired)

            if parsed is not None:
                return parsed

            # 5. Salvage complete objects from truncated text.
            #    Runs BEFORE raw_decode(repaired) so we recover ALL complete
            #    items from a truncated list, not just the first one.
            salvaged = ParserHelper._salvage_objects(repaired)

            if salvaged:
                return salvaged

            salvaged = ParserHelper._salvage_objects(candidate)

            if salvaged:
                return salvaged

            # 6. Raw decode on repaired text as a last fallback.
            parsed = ParserHelper._raw_decode(repaired)

            if parsed is not None:
                return parsed

        return None

    @staticmethod
    def parse_json_text(raw_text: Any) -> Optional[Union[dict, list]]:
        """
        Parses raw text and then unwraps known payload envelopes.

        Handles:
        - direct JSON
        - pseudo-JSON
        - markdown-wrapped JSON
        - truncated JSON via salvage
        - NovaVision output param envelopes
        - OpenAI/Claude style text envelopes
        """
        parsed = ParserHelper._parse_json_text_raw(raw_text)

        if parsed is None:
            return None

        return ParserHelper.unwrap_payload(parsed)
    
    @staticmethod
    def _loads(text: str) -> Any:
        try:
            return json.loads(text)
        except Exception:
            return None

    @staticmethod
    def _raw_decode(text: str) -> Any:
        decoder = json.JSONDecoder()

        for start_char in ["{", "["]:
            start = text.find(start_char)

            while start != -1:
                try:
                    obj, _ = decoder.raw_decode(text[start:])
                    return obj
                except Exception:
                    start = text.find(start_char, start + 1)

        return None

    @staticmethod
    def unwrap_payload(parsed_data: Any, depth: int = 0) -> Any:
        """
        Unwraps known payload envelopes and returns the actual model payload.

        Supported envelopes:
        - NovaVision output param:
            {
                "name": "outputText",
                "value": "...",
                "type": "string"
            }

        - OpenAI:
            {
                "choices": [
                    {
                        "message": {
                            "content": "..."
                        }
                    }
                ]
            }

        - Claude:
            {
                "content": [
                    {
                        "type": "text",
                        "text": "..."
                    }
                ]
            }

        - Generic:
            {
                "outputText": "..."
            }
            {
                "output": "..."
            }
            {
                "text": "..."
            }
        """
        if parsed_data is None or depth > 4:
            return parsed_data

        if isinstance(parsed_data, list):
            return parsed_data

        if not isinstance(parsed_data, dict):
            return parsed_data

        # NovaVision output param envelope.
        if "value" in parsed_data and any(
            key in parsed_data
            for key in ["name", "type", "listen", "branch"]
        ):
            inner_value = parsed_data["value"]

            if isinstance(inner_value, str):
                inner_parsed = ParserHelper._parse_json_text_raw(inner_value)

                if inner_parsed is not None:
                    return ParserHelper.unwrap_payload(
                        inner_parsed,
                        depth + 1
                    )

                return parsed_data

            return ParserHelper.unwrap_payload(
                inner_value,
                depth + 1
            )

        # OpenAI style envelope.
        choices = parsed_data.get("choices")

        if isinstance(choices, list) and choices:
            first_choice = choices[0]

            if isinstance(first_choice, dict):
                message = first_choice.get("message")

                content = None

                if isinstance(message, dict):
                    content = message.get("content")
                elif "text" in first_choice:
                    content = first_choice.get("text")

                if isinstance(content, str):
                    inner_parsed = ParserHelper._parse_json_text_raw(content)

                    if inner_parsed is not None:
                        return ParserHelper.unwrap_payload(
                            inner_parsed,
                            depth + 1
                        )

        # Claude style envelope.
        content = parsed_data.get("content")

        if isinstance(content, list):
            text_parts = []

            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    text_parts.append(item["text"])

            if text_parts:
                combined_text = "\n".join(text_parts)

                inner_parsed = ParserHelper._parse_json_text_raw(combined_text)

                if inner_parsed is not None:
                    return ParserHelper.unwrap_payload(
                        inner_parsed,
                        depth + 1
                    )

        # Generic text wrappers.
        for key in ["outputText", "output", "text", "result", "data"]:
            if key not in parsed_data:
                continue

            value = parsed_data[key]

            if isinstance(value, str):
                inner_parsed = ParserHelper._parse_json_text_raw(value)

                if inner_parsed is not None and isinstance(inner_parsed, (dict, list)):
                    return ParserHelper.unwrap_payload(
                        inner_parsed,
                        depth + 1
                    )

            elif isinstance(value, (dict, list)):
                return ParserHelper.unwrap_payload(
                    value,
                    depth + 1
                )

        return parsed_data
    
    @staticmethod
    def _repair_json_like_text(text: str) -> str:
        """
        Repairs common LLM/VLM pseudo-JSON outputs.

        Supports:
        - unquoted keys
        - unquoted string values
        - trailing commas
        - simple object/array structures
        """
        text = text.strip()

        if not text:
            return text

        # Normalize literal escaped sequences coming from serialized payloads.
        text = text.replace(r"\n", "\n")
        text = text.replace(r"\r", "\n")
        text = text.replace(r"\t", "\t")

        # Remove trailing commas.
        text = re.sub(
            r",\s*([}\]])",
            r"\1",
            text
        )

        # Quote unquoted object keys.
        text = re.sub(
            r'([{,]\s*)([\w$\-\.]+)\s*:',
            r'\1"\2":',
            text
        )

        # Quote unquoted keys at line start.
        text = re.sub(
            r'^\s*([\w$\-\.]+)\s*:',
            r'"\1":',
            text,
            flags=re.MULTILINE
        )

        def is_raw_literal(value: str) -> bool:
            value = value.strip()

            if not value:
                return False

            if value.lower() in {"true", "false", "null"}:
                return True

            if re.fullmatch(
                r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
                value
            ):
                return True

            return False

        def quote_value(match):
            prefix = match.group(1)
            value = match.group(2).strip()

            if is_raw_literal(value):
                return prefix + value

            return prefix + json.dumps(value, ensure_ascii=False)

        # Quote unquoted string values after colon.
        text = re.sub(
            r'(:\s*)([^\s"\'{\[\],:][^,\]\}\n]*?)\s*(?=[,\]\}\n]|$)',
            quote_value,
            text,
            flags=re.MULTILINE
        )

        return text

    @staticmethod
    def _repair_and_load(text: str) -> Any:
        """
        Repairs a small JSON-like block and tries to parse it.
        """
        repaired = ParserHelper._repair_json_like_text(text)

        parsed = ParserHelper._loads(repaired)

        if parsed is not None:
            return parsed

        return ParserHelper._raw_decode(repaired)

    @staticmethod
    def _find_balanced_object_end(text: str, start: int) -> Optional[int]:
        """
        Finds the closing brace index of a balanced {...} object.

        Returns:
        - end index when object is complete
        - None when object is truncated or unbalanced
        """
        if start >= len(text) or text[start] != "{":
            return None

        depth = 0
        in_string = False
        escape = False

        for index in range(start, len(text)):
            char = text[index]

            if escape:
                escape = False
                continue

            if char == "\\":
                escape = True
                continue

            if char == '"':
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == "{":
                depth += 1

            elif char == "}":
                depth -= 1

                if depth == 0:
                    return index

        return None

    @staticmethod
    def _salvage_objects(text: str) -> Optional[List[dict]]:
        """
        Salvages complete {...} objects from truncated or broken text.

        This is useful when a VLM output is cut off due to token limits.

        Example broken input:
            {detections: [{x_min: 0.1, class_name: person, confidence: 0.9}, {x_min: 0.2, y_

        Salvage result:
            [
                {
                    "x_min": 0.1,
                    "class_name": "person",
                    "confidence": 0.9
                }
            ]
        """
        if not text:
            return None

        objects = []
        length = len(text)
        index = 0

        while index < length:
            if text[index] != "{":
                index += 1
                continue

            end = ParserHelper._find_balanced_object_end(
                text=text,
                start=index
            )

            if end is None:
                index += 1
                continue

            object_text = text[index:end + 1]

            parsed_object = ParserHelper._repair_and_load(object_text)

            if isinstance(parsed_object, dict) and parsed_object:
                objects.append(parsed_object)
                index = end + 1
            else:
                index += 1

        if objects:
            return objects

        return None

    @staticmethod
    def _is_box_dict(d: Any) -> bool:
        """
        Returns True when the dict itself holds box coordinates directly,
        i.e. it is a flat single detection.

        Examples:
            {x_min, y_min, x_max, y_max}
            {x, y, width, height}
            {cx, cy, width, height}
        """
        if not isinstance(d, dict):
            return False

        xyxy = [
            ["x_min", "xmin", "x1", "left", "x"],
            ["y_min", "ymin", "y1", "top", "y"],
            ["x_max", "xmax", "x2", "right"],
            ["y_max", "ymax", "y2", "bottom"]
        ]

        if ParserHelper._get_four_from_dict(d, xyxy):
            return True

        xywh = [
            ["x", "left"],
            ["y", "top"],
            ["width", "w"],
            ["height", "h"]
        ]

        if ParserHelper._get_four_from_dict(d, xywh):
            return True

        cxcywh = [
            ["cx", "center_x"],
            ["cy", "center_y"],
            ["width", "w"],
            ["height", "h"]
        ]

        if ParserHelper._get_four_from_dict(d, cxcywh):
            return True

        return False
    
    @staticmethod
    def parse_expected_fields(raw_fields: Any) -> List[str]:
        """
        Parses expected fields input.

        Supported formats:
        - name,age,result
        - employees[0].firstName,employees[1].lastName
        - employees[*].firstName
        - ["employees[0].firstName", "employees[1].lastName"]
        """
        if raw_fields is None:
            return []

        if isinstance(raw_fields, list):
            return [
                str(item).strip()
                for item in raw_fields
                if str(item).strip()
            ]

        if isinstance(raw_fields, dict):
            return [
                str(key).strip()
                for key in raw_fields.keys()
                if str(key).strip()
            ]

        text = str(raw_fields).strip()

        if not text:
            return []

        # Only parse as JSON when the entire string is a JSON list/object.
        if text.startswith("[") or text.startswith("{"):
            try:
                parsed = json.loads(text)

                if isinstance(parsed, list):
                    return [
                        str(item).strip()
                        for item in parsed
                        if str(item).strip()
                    ]

                if isinstance(parsed, dict):
                    return [
                        str(key).strip()
                        for key in parsed.keys()
                        if str(key).strip()
                    ]

            except Exception:
                pass

        return [
            item.strip()
            for item in text.split(",")
            if item.strip()
        ]

    @staticmethod
    def parse_classes(raw_classes: Any) -> List[str]:
        """
        Parses class list from comma-separated string, JSON list, or JSON object.
        """
        if raw_classes is None:
            return []

        if isinstance(raw_classes, list):
            classes = []

            for item in raw_classes:
                if isinstance(item, dict):
                    class_name = ParserHelper.get_string(item, ParserHelper.CLASS_KEYS)
                    if class_name:
                        classes.append(class_name)
                elif str(item).strip():
                    classes.append(str(item).strip())

            return classes

        if isinstance(raw_classes, dict):
            return [
                str(key).strip()
                for key in raw_classes.keys()
                if str(key).strip()
            ]

        text = str(raw_classes).strip()

        if not text:
            return []

        parsed = ParserHelper.parse_json_text(text)

        if isinstance(parsed, list):
            classes = []

            for item in parsed:
                if isinstance(item, dict):
                    class_name = ParserHelper.get_string(item, ParserHelper.CLASS_KEYS)
                    if class_name:
                        classes.append(class_name)
                elif str(item).strip():
                    classes.append(str(item).strip())

            return classes

        if isinstance(parsed, dict):
            return [
                str(key).strip()
                for key in parsed.keys()
                if str(key).strip()
            ]

        return [
            item.strip()
            for item in text.split(",")
            if item.strip()
        ]

    @staticmethod
    def parse_path(path: str) -> List[Tuple[str, Any]]:
        """
        Parses nested path strings into token list.

        Examples:
        employees
        employees[0].firstName
        employees[*].firstName
        employees.firstName
        """
        tokens = []

        if not isinstance(path, str):
            return tokens

        path = path.strip()

        if not path:
            return tokens

        pattern = re.compile(r"([^\.\[\]]+)|\[(.*?)\]")

        for match in pattern.finditer(path):
            if match.group(1) is not None:
                key = match.group(1).strip()

                if key:
                    tokens.append(("key", key))
            else:
                inner = match.group(2).strip()

                if inner == "*":
                    tokens.append(("wildcard", None))
                else:
                    try:
                        tokens.append(("index", int(inner)))
                    except Exception:
                        inner = inner.strip("'\"")

                        if inner:
                            tokens.append(("key", inner))

        return tokens

    @staticmethod
    def _resolve_path(data: Any, tokens: List[Tuple[str, Any]]) -> Tuple[Any, bool]:
        """
        Recursively resolves parsed path tokens against JSON data.

        Returns:
        value, complete

        complete is True when the full path is resolved successfully.
        complete is False when the path or a part of it cannot be resolved.
        """
        if not tokens:
            return data, True

        token_type, token_value = tokens[0]
        rest = tokens[1:]

        if token_type == "key":
            key = token_value

            if isinstance(data, dict):
                if key in data:
                    return ParserHelper._resolve_path(data[key], rest)

                return None, False

            if isinstance(data, list):
                if len(data) == 0:
                    return [], True

                values = []
                found_any = False
                complete = True

                for item in data:
                    value, ok = ParserHelper._resolve_path(item, tokens)

                    if ok:
                        found_any = True
                        values.append(value)
                    else:
                        complete = False
                        values.append(None)

                if found_any:
                    return values, complete

                return values, False

            return None, False

        if token_type == "index":
            index = token_value

            if isinstance(data, list) and len(data) > 0:
                if -len(data) <= index < len(data):
                    return ParserHelper._resolve_path(data[index], rest)

            return None, False

        if token_type == "wildcard":
            if isinstance(data, list):
                if len(data) == 0:
                    return [], True

                values = []
                found_any = False
                complete = True

                for item in data:
                    value, ok = ParserHelper._resolve_path(item, rest)

                    if ok:
                        found_any = True
                        values.append(value)
                    else:
                        complete = False
                        values.append(None)

                if found_any:
                    return values, complete

                return values, False

            return None, False

        return None, False

    @staticmethod
    def get_nested_value(data: Any, path: str) -> Tuple[Any, bool]:
        """
        Gets nested value from parsed JSON using path.

        If path is a simple field name and the field is not found at root level,
        performs controlled recursive search for that field name.

        Examples:
        employees
        employees[0].firstName
        employees[*].firstName
        employees.firstName
        firstName
        """
        tokens = ParserHelper.parse_path(path)

        if not tokens:
            return None, False

        if len(tokens) == 1 and tokens[0][0] == "key":
            key = tokens[0][1]

            if isinstance(data, dict) and key in data:
                return data[key], True

            return ParserHelper.find_key(data, key)

        return ParserHelper._resolve_path(data, tokens)

    @staticmethod
    def find_key(
        data: Any,
        key: str,
        max_depth: int = 10,
        max_matches: int = 100,
        max_nodes: int = 10000
    ) -> Tuple[Any, bool]:
        """
        Searches key recursively inside parsed JSON data.

        Stop rules:
        - Do not descend into the value of a matched key.
        - Stop when max_depth is exceeded.
        - Stop when max_matches is reached.
        - Stop when max_nodes is exceeded.

        Returns:
        value, complete

        If exactly one match is found, returns the value.
        If multiple matches are found, returns a list of values.
        If search is truncated by limits, complete is False.
        """
        matches = []

        state = {
            "nodes": 0,
            "stopped": False
        }

        def _search(current: Any, depth: int):
            if state["stopped"]:
                return

            if depth > max_depth or state["nodes"] > max_nodes:
                state["stopped"] = True
                return

            if len(matches) >= max_matches:
                state["stopped"] = True
                return

            if isinstance(current, dict):
                state["nodes"] += 1

                if key in current:
                    matches.append(current[key])
                    return

                for value in current.values():
                    _search(value, depth + 1)

                    if state["stopped"]:
                        return

            elif isinstance(current, list):
                state["nodes"] += 1

                for item in current:
                    _search(item, depth + 1)

                    if state["stopped"]:
                        return

        _search(data, 0)

        if not matches:
            return None, False

        complete = not state["stopped"]

        if len(matches) == 1:
            return matches[0], complete

        return matches, complete

    @staticmethod
    def build_class_map(classes: List[str]) -> Dict[str, int]:
        return {
            str(class_name): index
            for index, class_name in enumerate(classes or [])
        }

    @staticmethod
    def build_normalized_class_map(classes: List[str]) -> Dict[str, int]:
        normalized_map = {}

        for index, class_name in enumerate(classes or []):
            normalized_name = ParserHelper.normalize_class_name(class_name)

            if normalized_name and normalized_name not in normalized_map:
                normalized_map[normalized_name] = index

        return normalized_map

    @staticmethod
    def normalize_class_name(class_name: Any) -> str:
        if class_name is None:
            return ""

        name = str(class_name).strip().lower()

        name = re.sub(
            r"[_\-]+",
            " ",
            name
        )

        name = re.sub(
            r"\s+",
            " ",
            name
        )

        return name

    @staticmethod
    def resolve_class_id(
        class_name: Any,
        class_list: List[str],
        class_map: Dict[str, int],
        normalized_class_map: Dict[str, int],
        auto_when_empty: bool = True
    ) -> Tuple[str, int]:
        """
        Resolves class name to class ID.

        Behavior:
        - If class_list is empty and auto_when_empty is True:
            return deterministic generated class ID.
        - If class_list is empty and auto_when_empty is False:
            return -1.
        - If class_list is not empty:
            first try exact match, then normalized match.
            If no match found, return -1.
        """
        original_name = str(class_name or "").strip()

        if not original_name:
            return "", -1

        if not class_list:
            if auto_when_empty:
                return original_name, ParserHelper.generate_class_id(original_name)

            return original_name, -1

        if original_name in class_map:
            index = class_map[original_name]
            return str(class_list[index]), int(index)

        normalized_name = ParserHelper.normalize_class_name(original_name)

        if normalized_name in normalized_class_map:
            index = normalized_class_map[normalized_name]
            return str(class_list[index]), int(index)

        return original_name, -1

    @staticmethod
    def clamp(value: Any, min_value: float = 0.0, max_value: float = 1.0) -> float:
        try:
            value = float(value)
        except Exception:
            return min_value

        if value < min_value:
            return min_value

        if value > max_value:
            return max_value

        return value

    @staticmethod
    def generate_class_id(class_name: str) -> int:
        return zlib.crc32(str(class_name).encode("utf-8")) % 1000000

    @staticmethod
    def get_string(item: Any, keys: List[str], default: Optional[str] = None) -> Optional[str]:
        if not isinstance(item, dict):
            return default

        for key in keys:
            if key in item and item[key] is not None:
                return str(item[key])

        return default

    @staticmethod
    def parse_confidence(item: Any, default: float = 0.0) -> float:
        if not isinstance(item, dict):
            return ParserHelper.clamp(default)

        for key in ParserHelper.CONFIDENCE_KEYS:
            if key in item and item[key] is not None:
                try:
                    value = float(item[key])
                except Exception:
                    continue

                if value > 1.0 and value <= 100.0:
                    value = value / 100.0

                return ParserHelper.clamp(value)

        return ParserHelper.clamp(default)

    @staticmethod
    def _get_dict_value(data: dict, aliases: List[str]) -> Any:
        for alias in aliases:
            if alias in data and data[alias] is not None:
                return data[alias]

        return None

    @staticmethod
    def _get_four_from_dict(data: dict, groups: List[List[str]]) -> Optional[List[float]]:
        values = []

        for aliases in groups:
            value = ParserHelper._get_dict_value(data, aliases)

            if value is None:
                return None

            try:
                values.append(float(value))
            except Exception:
                return None

        return values

    @staticmethod
    def extract_detection_items(parsed_data: Any) -> Optional[List[Any]]:
        """
        Finds detection item list from parsed JSON.
        Returns None when detection format cannot be determined.
        Returns empty list when format is valid but there are no detections.
        """
        if parsed_data is None:
            return None

        if isinstance(parsed_data, list):
            return parsed_data

        if isinstance(parsed_data, dict):
            if isinstance(parsed_data.get("responses"), list) and parsed_data["responses"]:
                return ParserHelper.extract_detection_items(parsed_data["responses"][0])

            for key in ParserHelper.DETECTION_LIST_KEYS:
                value = parsed_data.get(key)

                if isinstance(value, list):
                    return value

                if isinstance(value, dict):
                    for nested_key in ParserHelper.DETECTION_LIST_KEYS:
                        nested_value = value.get(nested_key)

                        if isinstance(nested_value, list):
                            return nested_value

            if any(key in parsed_data for key in ParserHelper.BOX_KEYS):
                return [parsed_data]

            # Flat single detection: coordinates live directly inside the dict.
            if ParserHelper._is_box_dict(parsed_data):
                return [parsed_data]

            for value in parsed_data.values():
                if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
                    return value

            return None

        return None

    @staticmethod
    def _extract_vertices_box(vertices: Any) -> Optional[List[float]]:
        """
        Extracts box from vertices or normalizedVertices structure.

        Returns:
        [x_min, y_min, x_max, y_max]
        """
        if not isinstance(vertices, list) or len(vertices) < 2:
            return None

        xs = []
        ys = []

        for point in vertices:
            if not isinstance(point, dict):
                continue

            x = point.get("x")
            y = point.get("y")

            if x is None or y is None:
                continue

            try:
                xs.append(float(x))
                ys.append(float(y))
            except Exception:
                continue

        if not xs or not ys:
            return None

        return [
            min(xs),
            min(ys),
            max(xs),
            max(ys)
        ]
    
    @staticmethod
    def extract_box_with_source(item: Any) -> Optional[Tuple[List[float], str]]:
        """
        Extracts box as [x_min, y_min, x_max, y_max] and returns source key.
        """
        if not isinstance(item, dict):
            return None

        candidates = []

        for key in ParserHelper.BOX_KEYS:
            if key in item and item[key] is not None:
                candidates.append((key, item[key]))

        if not candidates:
            candidates.append(("item", item))

        for key, box in candidates:
            if isinstance(box, dict):
                # Direct vertices support.
                for vertices_key in ["normalizedVertices", "vertices"]:
                    if vertices_key in box:
                        vertices_box = ParserHelper._extract_vertices_box(box[vertices_key])

                        if vertices_box:
                            return vertices_box, f"{key}.{vertices_key}"

                # Nested boundingPoly support.
                if "boundingPoly" in box and isinstance(box["boundingPoly"], dict):
                    bounding_poly = box["boundingPoly"]

                    for vertices_key in ["normalizedVertices", "vertices"]:
                        if vertices_key in bounding_poly:
                            vertices_box = ParserHelper._extract_vertices_box(
                                bounding_poly[vertices_key]
                            )

                            if vertices_box:
                                return vertices_box, f"{key}.boundingPoly.{vertices_key}"

                values = ParserHelper._get_four_from_dict(
                    box,
                    [
                        ["x_min", "xmin", "x1", "left", "x"],
                        ["y_min", "ymin", "y1", "top", "y"],
                        ["x_max", "xmax", "x2", "right"],
                        ["y_max", "ymax", "y2", "bottom"]
                    ]
                )

                if values:
                    return values, key

                values = ParserHelper._get_four_from_dict(
                    box,
                    [
                        ["x", "left"],
                        ["y", "top"],
                        ["width", "w"],
                        ["height", "h"]
                    ]
                )

                if values:
                    x, y, w, h = values
                    return [x, y, x + w, y + h], key

                values = ParserHelper._get_four_from_dict(
                    box,
                    [
                        ["cx", "center_x"],
                        ["cy", "center_y"],
                        ["width", "w"],
                        ["height", "h"]
                    ]
                )

                if values:
                    cx, cy, w, h = values
                    return [
                        cx - (w / 2.0),
                        cy - (h / 2.0),
                        cx + (w / 2.0),
                        cy + (h / 2.0)
                    ], key

            if isinstance(box, (list, tuple)) and len(box) >= 4:
                try:
                    values = [float(v) for v in box[:4]]
                except Exception:
                    continue

                if key == "box_2d":
                    return [values[1], values[0], values[3], values[2]], key

                return values, key

        return None

    @staticmethod
    def extract_box(item: Any) -> Optional[List[float]]:
        """
        Extracts box as [x_min, y_min, x_max, y_max].
        """
        result = ParserHelper.extract_box_with_source(item)

        if result is None:
            return None

        box, _ = result
        return box

    @staticmethod
    def normalize_box(
        box: Optional[List[float]],
        width: int,
        height: int,
        source_key: Optional[str] = None,
        coordinate_format: str = "auto",
        model_type: Optional[str] = None
    ) -> Optional[List[float]]:
        """
        Converts coordinates to pixel coordinates.

        Supported coordinate formats:
        - auto
        - normalized-0-1
        - normalized-0-1000
        - pixel

        Returns [left, top, width, height].
        """
        if box is None or width <= 0 or height <= 0:
            return None

        try:
            x1, y1, x2, y2 = [float(v) for v in box]
        except Exception:
            return None

        coordinate_format = str(coordinate_format or "auto").lower()
        model_type = str(model_type or "").lower()

        max_value = max(abs(x1), abs(y1), abs(x2), abs(y2))
        max_dimension = max(float(width), float(height))

        def apply_0_1(values):
            bx1, by1, bx2, by2 = values
            return bx1 * width, by1 * height, bx2 * width, by2 * height

        def apply_0_1000(values):
            bx1, by1, bx2, by2 = values
            return (
                (bx1 / 1000.0) * width,
                (by1 / 1000.0) * height,
                (bx2 / 1000.0) * width,
                (by2 / 1000.0) * height
            )

        scaled = False

        if coordinate_format == "normalized-0-1":
            x1, y1, x2, y2 = apply_0_1((x1, y1, x2, y2))
            scaled = True

        elif coordinate_format == "normalized-0-1000":
            x1, y1, x2, y2 = apply_0_1000((x1, y1, x2, y2))
            scaled = True

        elif coordinate_format == "pixel":
            scaled = True

        else:
            # Auto mode.
            if max_value <= 1.05:
                x1, y1, x2, y2 = apply_0_1((x1, y1, x2, y2))
                scaled = True

            elif source_key == "box_2d" and max_value <= 1000.0:
                x1, y1, x2, y2 = apply_0_1000((x1, y1, x2, y2))
                scaled = True

            elif model_type in ["google-gemini", "gemini"] and max_value <= 1000.0:
                x1, y1, x2, y2 = apply_0_1000((x1, y1, x2, y2))
                scaled = True

            elif max_value > max_dimension and max_value <= 1000.0:
                x1, y1, x2, y2 = apply_0_1000((x1, y1, x2, y2))
                scaled = True

        if x2 < x1:
            x1, x2 = x2, x1

        if y2 < y1:
            y1, y2 = y2, y1

        x1 = max(0.0, min(float(width), x1))
        y1 = max(0.0, min(float(height), y1))
        x2 = max(0.0, min(float(width), x2))
        y2 = max(0.0, min(float(height), y2))

        w = x2 - x1
        h = y2 - y1

        if w <= 0.0 or h <= 0.0:
            return None

        return [x1, y1, w, h]
        
    @staticmethod
    def detect_classification_format(parsed_data: Any) -> str:
        """
        Returns:
        - "single"
        - "multi"
        - "unknown"
        """
        if isinstance(parsed_data, dict):
            if "predicted_classes" in parsed_data and isinstance(parsed_data["predicted_classes"], list):
                return "multi"

            if "class_name" in parsed_data and "confidence" in parsed_data:
                return "single"

            if ParserHelper.get_string(parsed_data, ParserHelper.CLASS_KEYS) is not None:
                return "single"

        if isinstance(parsed_data, list):
            return "multi"

        return "unknown"

    @staticmethod
    def extract_classification_items(parsed_data: Any) -> List[Dict[str, Any]]:
        """
        Extracts classification items and merges duplicate classes by keeping
        the maximum confidence value.
        """
        raw_items = []

        if isinstance(parsed_data, dict):
            if "predicted_classes" in parsed_data and isinstance(parsed_data["predicted_classes"], list):
                raw_items = parsed_data["predicted_classes"]
            elif "predictions" in parsed_data and isinstance(parsed_data["predictions"], list):
                raw_items = parsed_data["predictions"]
            elif "class_name" in parsed_data or ParserHelper.get_string(parsed_data, ParserHelper.CLASS_KEYS):
                raw_items = [parsed_data]
            else:
                for value in parsed_data.values():
                    if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
                        raw_items = value
                        break

        elif isinstance(parsed_data, list):
            raw_items = parsed_data

        items = []

        for item in raw_items:
            if isinstance(item, dict):
                class_name = ParserHelper.get_string(item, ParserHelper.CLASS_KEYS)

                if class_name is None:
                    continue

                confidence = ParserHelper.parse_confidence(item, default=0.0)

                items.append({
                    "class_name": class_name,
                    "confidence": confidence
                })

            elif isinstance(item, str) and item.strip():
                items.append({
                    "class_name": item.strip(),
                    "confidence": 1.0
                })

        merged = {}

        for item in items:
            class_name = item["class_name"]

            if class_name not in merged:
                merged[class_name] = item
            elif item["confidence"] > merged[class_name]["confidence"]:
                merged[class_name] = item

        return list(merged.values())
