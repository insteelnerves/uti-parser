# **Parser for NovaVision**

**Parser** is a VLM/LLM output parsing package developed for the NovaVision system. It converts raw text outputs from various VLM/LLM models (Google Gemini, OpenAI GPT, Anthropic Claude, GCP Vision, Qwen AI, Kimi AI, Florence-2, SpaceXAI) into standardized NovaVision Detection and Classification formats.

> With this package, you can seamlessly integrate any VLM/LLM model into your NovaVision pipeline without worrying about inconsistent JSON formats, truncated outputs, or class ID mapping.

---

## **✨ Key Features**

- ✅ **Pseudo-JSON Repair** — Automatically fixes VLM outputs with unquoted keys and values
- ✅ **Salvage Mode** — Recovers complete detections from truncated outputs (token limit cuts)
- ✅ **Envelope Unwrap** — Handles NovaVision, OpenAI, and Claude response wrappers
- ✅ **Auto Class ID** — Generates deterministic hash-based IDs when no class list is provided
- ✅ **Configurable Class Mapping** — Maps user-defined class lists to stable IDs
- ✅ **Multi-Format Support** — Works with strict JSON, markdown-wrapped JSON, pseudo-JSON, and structured lists

---

## **📦 Package Structure**

The package consists of three executors:

---

### **1. JsonParser**

The **JsonParser executor** extracts specific fields from raw JSON or pseudo-JSON text outputs.

**Input:** `inputRawText` (string, dict, or list)

**Process Flow:**

1. Raw text is parsed using strict JSON parser
2. If strict parse fails, pseudo-JSON repair is applied
3. If repair fails, salvage mode extracts complete objects
4. Envelope unwrap is applied to known wrappers
5. Expected fields are extracted using nested path resolution
6. Extracted data is returned as `outputData`

**Configuration Parameters:**

- **Expected Fields**: Comma-separated list of fields to extract
  ```text
  name,age,product.price,employees[*].firstName
  ```

  Supports:
  - Simple fields: `name,age`
  - Nested paths: `employees[0].firstName`
  - Wildcards: `employees[*].firstName`
  - Dot notation: `employees.firstName` (implicit wildcard)

**Output Example:**

```json
{
    "name": "outputData",
    "value": {
        "name": "John Doe",
        "age": 30,
        "employees[*].firstName": ["Alice", "Bob", "Charlie"]
    },
    "type": "object"
}
```

---

### **2. VLMAsDetector**

The **VLMAsDetector executor** converts VLM/LLM detection outputs into NovaVision's standardized `Detection` format with bounding boxes, confidence scores, and class IDs.

**Input:** `inputImage`, `inputRawText` (string, dict, or list)

**Process Flow:**

1. Image is loaded via `Image.get_frame()` to get dimensions
2. Raw text is parsed (strict → repair → salvage pipeline)
3. Envelope unwrap extracts actual detection data
4. Detection items are extracted from various formats
5. Bounding boxes are extracted and normalized to pixel coordinates
6. Class IDs are resolved using config mapping or auto-generated
7. NovaVision `Detection` objects are created
8. Results are returned as `outputDetections`

<img width="997" height="312" alt="VLMAsDetector Flow" src="https://github.com/user-attachments/assets/placeholder-detector-flow" />

**Configuration Parameters:**

- **Model Type**: VLM/LLM model that produced the output
  - `auto` (default) — Auto-detect format
  - `openai` — OpenAI GPT models
  - `google-gemini` — Google Gemini models
  - `anthropic-claude` — Anthropic Claude models
  - `florence-2` — Microsoft Florence-2
  - `gcp-vision` — Google Cloud Vision API
  - `qwen-ai` — Alibaba Qwen models
  - `kimi-ai` — Moonshot Kimi models
  - `spacexai` — SpaceXAI (Grok) models

- **Task Type**: Detection task type
  - `object-detection` (default) — Standard object detection
  - `open-vocabulary-object-detection` — Open vocabulary detection
  - `object-detection-and-caption` — Detection with captions
  - `phrase-grounded-object-detection` — Phrase-based grounding
  - `region-proposal` — Class-agnostic region proposals
  - `ocr-with-text-detection` — Text detection and OCR

- **Coordinate Format**: Bounding box coordinate format
  - `auto` (default) — Auto-detect from values
  - `normalized-0-1` — Coordinates in [0, 1] range
  - `normalized-0-1000` — Coordinates in [0, 1000] range
  - `pixel` — Absolute pixel coordinates

- **Classes** (Optional): Comma-separated class list for ID mapping
  ```text
  person,car,dog,cat
  ```
  - If provided: known classes get config index, unknown get `-1`
  - If empty: deterministic hash-based auto IDs are generated

**Output Example:**

```json
{
    "name": "outputDetections",
    "value": [
        {
            "boundingBox": {
                "left": 174.2,
                "top": 254.27,
                "width": 105.3,
                "height": 453.04
            },
            "confidence": 0.9,
            "classLabel": "person",
            "classId": 886774
        },
        {
            "boundingBox": {
                "left": 226.2,
                "top": 318.84,
                "width": 119.6,
                "height": 438.92
            },
            "confidence": 0.89,
            "classLabel": "person",
            "classId": 886774
        }
    ],
    "type": "list"
}
```

**Supported Input Formats:**

**Gemini pseudo-JSON:**
```javascript
{
  detections: [
    {x_min: 0.134, y_min: 0.252, x_max: 0.215, y_max: 0.701, class_name: person, confidence: 0.9}
  ]
}
```

**GCP Vision structured list:**
```json
[
  {
    "boundingBox": {"left": 255, "top": 439, "width": 75, "height": 112},
    "confidence": 0.789,
    "classLabel": "Top",
    "classId": -1
  }
]
```

**OpenAI envelope:**
```json
{
  "name": "outputText",
  "value": "{detections: [...]}",
  "type": "string"
}
```

---

### **3. VLMAsClassifier**

The **VLMAsClassifier executor** converts VLM/LLM classification outputs into NovaVision's standardized classification format (single-label or multi-label).

**Input:** `inputImage`, `inputRawText` (string, dict, or list)

**Process Flow:**

1. Image is loaded via `Image.get_frame()` to get dimensions
2. Raw text is parsed (strict → repair → salvage pipeline)
3. Envelope unwrap extracts actual classification data
4. Classification format is detected (single or multi-label)
5. Class items are extracted from various formats
6. Class IDs are resolved using config mapping or auto-generated
7. Results are returned as `outputData`

<img width="891" height="638" alt="VLMAsClassifier Flow" src="https://github.com/user-attachments/assets/placeholder-classifier-flow" />

**Configuration Parameters:**

- **Model Type**: VLM/LLM model that produced the output
  - `auto` (default) — Auto-detect format
  - `openai` — OpenAI GPT models
  - `google-gemini` — Google Gemini models
  - `anthropic-claude` — Anthropic Claude models
  - `gcp-vision` — Google Cloud Vision API
  - `qwen-ai` — Alibaba Qwen models
  - `kimi-ai` — Moonshot Kimi models

- **Classes** (Optional): Comma-separated class list for ID mapping
  ```text
  person,car,dog,cat
  ```
  - If provided: known classes get config index, unknown get `-1`
  - If empty: deterministic hash-based auto IDs are generated

**Output Examples:**

**Single-label classification:**

```json
{
    "name": "outputData",
    "value": {
        "type": "classification",
        "width": 1300,
        "height": 1009,
        "inference_id": "3158fc7d-c1e7-4fd6-a2e3-b4ad1ced79d9",
        "parent_id": "3158fc7d-c1e7-4fd6-a2e3-b4ad1ced79d9",
        "top": {
            "class_name": "group of people",
            "class_id": 250314,
            "confidence": 0.99
        },
        "confidence": 0.99,
        "predictions": [
            {
                "class_name": "group of people",
                "class_id": 250314,
                "confidence": 0.99
            }
        ]
    },
    "type": "object"
}
```

**Multi-label classification:**

```json
{
    "name": "outputData",
    "value": {
        "type": "multi-label-classification",
        "width": 1300,
        "height": 1009,
        "inference_id": "b5a0718e-9034-4ad8-a496-c0ceb55177ff",
        "parent_id": "b5a0718e-9034-4ad8-a496-c0ceb55177ff",
        "predicted_classes": [
            {
                "class_name": "person",
                "class_id": 886774,
                "confidence": 0.99
            },
            {
                "class_name": "clothing",
                "class_id": 5233,
                "confidence": 0.98
            },
            {
                "class_name": "man",
                "class_id": 263384,
                "confidence": 0.97
            }
        ],
        "predictions": {
            "predicted_classes": [...]
        }
    },
    "type": "object"
}
```

**Supported Input Formats:**

**Gemini pseudo-JSON:**
```javascript
{
  predicted_classes: [
    {class: person, confidence: 0.99},
    {class: clothing, confidence: 0.98}
  ]
}
```

**OpenAI JSON:**
```json
{
  "class_name": "dog",
  "confidence": 0.87
}
```

---

## **🔧 Class ID Mapping Behavior**

The package provides intelligent class ID mapping:

### **When Classes Config is Empty:**

Auto-generates deterministic hash-based IDs:

```text
person    → 886774
clothing  → 5233
man       → 263384
woman     → 24349
```

**Advantages:**
- ✅ Stable across frames (same class name → same ID)
- ✅ No configuration needed
- ✅ Works with any VLM output

### **When Classes Config is Provided:**

Uses config list index as class ID:

**Config:** `man,woman,person,clothing`

**Mapping:**
```text
man       → 0 (config index)
woman     → 1 (config index)
person    → 2 (config index)
clothing  → 3 (config index)
shirt     → -1 (unknown class)
```

**Advantages:**
- ✅ Predictable, small IDs
- ✅ Matches your application's class schema
- ✅ Unknown classes clearly marked with `-1`

### **Normalize Matching:**

Class names are normalized for flexible matching:

```text
Person == person == PERSON
T-Shirt == t-shirt == t shirt == t_shirt
Group Of People == group of people == group-of-people
```

---

## **🛠️ Advanced Features**

### **Pseudo-JSON Repair**

VLM models often output JavaScript-like pseudo-JSON instead of strict JSON. The parser automatically repairs these:

**Input (pseudo-JSON):**
```javascript
{
  detections: [
    {x_min: 0.15, y_min: 0.25, class_name: person, confidence: 0.9}
  ]
}
```

**Repaired (strict JSON):**
```json
{
  "detections": [
    {"x_min": 0.15, "y_min": 0.25, "class_name": "person", "confidence": 0.9}
  ]
}
```

**Repairs applied:**
- Unquoted keys → quoted keys
- Unquoted string values → quoted values
- Trailing commas → removed
- Literal `\n` escapes → actual newlines

---

### **Salvage Mode (Truncated Output Recovery)**

When VLM outputs are truncated due to token limits, the salvage algorithm recovers complete detections:

**Truncated input:**
```javascript
{detections: [{x_min: 0.133, y_min: 0.251, x_max: 0.209, y_max: 0.699, class_name: person, confidence: 0.94}, {x_min: 0.177, y_min: 0.296, x_max: 0.273, y_max:
```

**Salvage output:**
```json
[
  {
    "x_min": 0.133,
    "y_min": 0.251,
    "x_max": 0.209,
    "y_max": 0.699,
    "class_name": "person",
    "confidence": 0.94
  }
]
```

**How it works:**
1. Scans for balanced `{...}` objects
2. Extracts complete objects
3. Discards truncated (incomplete) objects
4. Returns list of recovered detections

---

### **Envelope Unwrap**

Automatically unwraps known response envelopes:

**NovaVision param envelope:**
```json
{
  "name": "outputText",
  "value": "{detections: [...]}",
  "type": "string",
  "listen": "continuous",
  "branch": "forward"
}
```

**OpenAI envelope:**
```json
{
  "choices": [
    {
      "message": {
        "content": "{detections: [...]}"
      }
    }
  ]
}
```

**Claude envelope:**
```json
{
  "content": [
    {
      "type": "text",
      "text": "{detections: [...]}"
    }
  ]
}
```

---

## **📊 Tested Models and Tasks**

### **Successfully Tested:**

| Model | Detection | Classification | Notes |
|-------|-----------|----------------|-------|
| Google Gemini | ✅ | ✅ | Pseudo-JSON repair works |
| OpenAI GPT-4o | ✅ | ✅ | Envelope unwrap works |
| GCP Vision | ✅ | N/A | Structured list input supported |

### **Not Tested:**

| Model | Reason |
|-------|--------|
| Anthropic Claude | API key not available |
| Alibaba Qwen | OpenRouter API key not available |
| Moonshot Kimi | OpenRouter API key not available |
| Florence-2 | Not available in NovaVision platform |
| SpaceXAI (Grok) | Not available in NovaVision platform |

### **Detection Tasks:**

| Task | Status | Notes |
|------|--------|-------|
| Object Detection | ✅ Tested | All models support this |
| Open Vocabulary Object Detection | ⚠️ Not tested | Models don't support this task |
| Object Detection and Caption | ⚠️ Not tested | Models don't support this task |
| Phrase Grounded Object Detection | ⚠️ Not tested | Models don't support this task |
| Region Proposal | ⚠️ Not tested | Models don't support this task |
| OCR with Text Detection | ⚠️ Not tested | Models don't support this task |

---

## **🔍 Fields Explanation**

### **Detection Output Fields:**

- **boundingBox**: Bounding box with `left`, `top`, `width`, `height` (pixel coordinates)
- **confidence**: Detection confidence score (0.0 to 1.0)
- **classLabel**: Class name (e.g., "person", "car")
- **classId**: Class ID (config index, auto hash, or -1 for unknown)
- **inference_id**: Unique UUID for this inference

### **Classification Output Fields:**

- **type**: `"classification"` (single) or `"multi-label-classification"` (multi)
- **width** / **height**: Image dimensions
- **inference_id**: Unique UUID for this inference
- **parent_id**: Parent inference ID (same as inference_id)
- **top**: Top prediction (single-label only)
- **confidence**: Top prediction confidence (single-label only)
- **predictions**: List of all predictions
- **predicted_classes**: Predicted classes (multi-label only)

### **Common Fields:**

- **class_name**: Class name from VLM output
- **class_id**: Resolved class ID
- **confidence**: Confidence score (0.0 to 1.0)

---

## **💡 Usage Tips**

### **For Best Results:**

1. **Use strict JSON prompts** when possible:
   ```text
   Return only valid JSON. Use double quotes for all keys and string values.
   ```

2. **Set reasonable token limits** to avoid truncation:
   - Detection: 4000-8000 tokens
   - Classification: 1000-2000 tokens

3. **Provide class list** for predictable IDs:
   ```text
   person,car,dog,cat,bicycle
   ```

4. **Use auto mode** for quick testing:
   - Model Type: `auto`
   - Coordinate Format: `auto`
   - Classes: (leave empty)

5. **Handle unknown classes** gracefully:
   - Filter out `class_id: -1` in downstream packages
   - Or add unknown classes to your config list

---

## **📝 Example Prompt for Object Detection**

```text
Detect all visible objects in the image.

Allowed classes:
person,car,dog,cat,bicycle

Rules:
- Use only the allowed class names
- Return at most 25 detections
- Use compact JSON format
- Coordinates must be normalized to [0, 1]

Return only JSON in this exact format:
{"detections":[{"class_name":"person","confidence":0.9,"x_min":0.1,"y_min":0.2,"x_max":0.4,"y_max":0.8}]}

If no object is found, return:
{"detections":[]}
```

---

## **📝 Example Prompt for Classification**

```text
Classify the main subject of the image into exactly one class.

Allowed classes:
person,car,dog,cat,bicycle

Rules:
- Return exactly one class
- Use the exact class name from the allowed list
- Confidence must be between 0 and 1

Return only JSON in this exact format:
{"class_name":"dog","confidence":0.87}
```

---

## **🚀 Quick Start**

1. **Add Parser package** to your NovaVision flow
2. **Connect VLM output** to Parser's `inputRawText`
3. **Select executor**: JsonParser, VLMAsDetector, or VLMAsClassifier
4. **Configure parameters**: Model type, task type, classes (optional)
5. **Run the flow** — Parser handles all format conversions automatically

---

## **📚 Documentation**

For detailed technical documentation, see [DOCUMENTATION.md](DOCUMENTATION.md)

---

## **🤝 Contributing**

This package is part of the NovaVision ecosystem. For issues, feature requests, or contributions, please refer to the NovaVision documentation.

---

**License:** See LICENSE file for details.
