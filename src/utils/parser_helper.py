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
        "text"
    ]

    CONFIDENCE_KEYS = [
        "confidence",
        "score",
        "prob",
        "probability",
        "conf"
    ]

    DETECTION_LIST_KEYS = [
        "detections",
        "predictions",
        "objects",
        "boxes",
        "results",
        "items",
        "data",
        "labels"
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
        "region"
    ]

    @staticmethod
    def new_inference_id() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def parse_json_text(raw_text: Any) -> Optional[Union[dict, list]]:
        """
        Parses raw JSON text or Markdown-wrapped JSON text.
        If multiple Markdown JSON blocks exist, only the first one is used.
        """
        if raw_text is None:
            return None

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

        decoder = json.JSONDecoder()

        for candidate in candidates:
            try:
                return json.loads(candidate)
            except Exception:
                pass

            for start_char in ["{", "["]:
                start = candidate.find(start_char)

                while start != -1:
                    try:
                        obj, _ = decoder.raw_decode(candidate[start:])
                        return obj
                    except Exception:
                        start = candidate.find(start_char, start + 1)

        return None

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
            return [str(key).strip() for key in raw_classes.keys() if str(key).strip()]

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
            return [str(key).strip() for key in parsed.keys() if str(key).strip()]

        return [item.strip() for item in text.split(",") if item.strip()]

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

        Examples:
        employees
        employees[0].firstName
        employees[*].firstName
        employees.firstName
        """
        tokens = ParserHelper.parse_path(path)

        if not tokens:
            return None, False

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
            for index, class_name in enumerate(classes)
        }

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
                    return ParserHelper.clamp(float(item[key]))
                except Exception:
                    pass

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

            for value in parsed_data.values():
                if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
                    return value

            return None

        return None

    @staticmethod
    def extract_box(item: Any) -> Optional[List[float]]:
        """
        Extracts box as [x_min, y_min, x_max, y_max].
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
                    return values

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
                    return [x, y, x + w, y + h]
                    
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
                    ]

            if isinstance(box, (list, tuple)) and len(box) >= 4:
                try:
                    values = [float(v) for v in box[:4]]
                except Exception:
                    continue
                    
                if key == "box_2d":
                    return [values[1], values[0], values[3], values[2]]

                return values

        return None

    @staticmethod
    def normalize_box(
        box: Optional[List[float]],
        width: int,
        height: int
    ) -> Optional[List[float]]:
        """
        Converts normalized 0-1 coordinates to pixel coordinates.
        Returns [left, top, width, height].
        """
        if box is None or width <= 0 or height <= 0:
            return None

        try:
            x1, y1, x2, y2 = [float(v) for v in box]
        except Exception:
            return None

        max_value = max(abs(x1), abs(y1), abs(x2), abs(y2))

        if max_value <= 1.0:
            x1 = x1 * width
            y1 = y1 * height
            x2 = x2 * width
            y2 = y2 * height

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
