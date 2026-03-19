# luxonis-eval

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [CLI](#cli)
  - [Python API Usage](#python-api-usage)
- [Architecture](#architecture)
  - [Key Base Classes](#key-base-classes)
  - [Evaluation Pipeline: Modular Design](#evaluation-pipeline-modular-design)
- [Configuration](#configuration)
  - [Task Name](#task-name)
  - [Data Loading \& Preprocessing](#data-loading--preprocessing)
  - [Output Parser](#output-parser)
  - [Evaluation Metrics](#evaluation-metrics)
  - [Visualization (Optional)](#visualization-optional)
  - [Inference Engine](#inference-engine)
  - [Full Example](#full-example)
- [Extending the Framework](#extending-the-framework)
  - [Adding a Custom DataLoader](#adding-a-custom-dataloader)
  - [Adding a Custom Engine](#adding-a-custom-engine)
  - [Adding a Custom Parser](#adding-a-custom-parser)
  - [Adding a Custom Metric](#adding-a-custom-metric)
  - [General Pattern](#general-pattern)
- [License](#license)

## Overview

**luxonis-eval** is a modular, extensible model evaluation framework designed to benchmark and evaluate neural network models across multiple inference backends. It supports running inference on Luxonis hardware devices (RVC2/RVC4) through DepthAI and on host through ONNX Runtime, computing standard quality metrics, and reporting throughput/latency performance.

The framework follows a **registry-based architecture** where each pluggable component (engines, dataloaders, parsers, metrics, and visualizers) is a self-registering module. This means you can swap, extend, or add any part of the pipeline without modifying the core evaluation logic. Just implement a new class inheriting from the respective base class, and it becomes available by name in your configuration.

## Features

- **Multiple Inference Backends**
  - [**DepthAI Engine**](luxonis_eval/engines/depthai_engine.py) – Run models exported as [NNArchive](https://docs.luxonis.com/software-v3/ai-inference/nn-archive) files on Luxonis devices via [DepthAI](https://docs.luxonis.com/software-v3/depthai/)
  - [**ONNX Engine**](luxonis_eval/engines/onnx_engine.py) – Run models on CPU/GPU using ONNX Runtime

- **DataLoaders**
  - [**LuxonisLoader**](https://github.com/luxonis/luxonis-ml/tree/main/luxonis_ml/data/loaders#luxonisml-loader) – Load datasets in a Luxonis Data Format (LDF)
  - [**BaseEvalLoader**](luxonis_eval/loaders/base_loader.py) – Base class for custom dataloaders

- **Supported Tasks**
  - `Classification` – Image classification
  - `Detection` – Bounding box detection
  - `SemanticSegmentation` – Per-pixel class labeling
  - `InstanceSegmentation` – Per-instance masks with detection
  - `KeypointDetection` – Body/object keypoint localization

- **Metrics**
  - [`TopKAccuracy`](luxonis_eval/metrics/topk_accuracy.py) – Top-1/Top-5 accuracy for classification
  - [`BboxMeanAveragePrecision`](luxonis_eval/metrics/bbox_map.py) – COCO-style mAP for bounding box detection
  - [`MaskMeanAveragePrecision`](luxonis_eval/metrics/mask_map.py) – COCO-style mAP for instance segmentation
  - [`KeypointMeanAveragePrecision`](luxonis_eval/metrics/keypoint_map.py) – OKS-based mAP for keypoint detection
  - [`MIoU`](luxonis_eval/metrics/mIoU.py) – Mean Intersection over Union for semantic segmentation
  - [`DiceCoefficient`](luxonis_eval/metrics/dice_coef.py) – Dice score for semantic segmentation
  - [`ThroughputMetric`](luxonis_eval/metrics/throughput.py) – Inference throughput and latency

- **Extensible Architecture** – Registry-based design using [`AutoRegisterMeta`](luxonis_eval/registry.py) for easy addition of custom engines, parsers, metrics, loaders, and visualizers

## Installation

```sh
pip install -e .
```

For development:

```sh
pip install -e ".[dev]"
```

## Usage

### CLI

```sh
# Run evaluation with a config file
python -m luxonis_eval eval --config path/to/config.yaml

# Run with CLI overrides
python -m luxonis_eval eval \
    --config path/to/config.yaml \
    --dataset-name coco \
    --model-path path/to/model.tar.xz \
    --backend depthai

# Using ONNX backend
python -m luxonis_eval eval \
    --config path/to/config.yaml \
    --dataset-name coco \
    --model-path path/to/model.onnx \
    --backend onnx

# Specify device IP for RVC4
python -m luxonis_eval eval \
    --config path/to/config.yaml \
    --device-ip 192.168.1.100
```

### Python API Usage

```python
from luxonis_eval.__main__ import eval_setup, eval_run
from luxonis_eval.utils.config import EvalConfig

# Load configuration
eval_cfg = EvalConfig.get_config(cfg="path/to/config.yaml")

# Setup engine and dataloader
infer_engine, dataloader = eval_setup(eval_cfg)

# Run evaluation
eval_run(eval_cfg, infer_engine, dataloader)
```

## Architecture

```bash
luxonis_eval/
├── engines/          # Inference backends
├── loaders/          # Dataset loaders
├── metrics/          # Evaluation metrics
├── parsers/          # Model output parsers
├── utils/            # Configuration, helper functions
├── visualizers/      # Result visualization
└── metadata/         # Class mapping files
```

### Key Base Classes

| Base Class | Location | Purpose |
| ---------- | -------- | ------- |
| [`BaseEngine`](luxonis_eval/engines/base_engine.py) | `engines/` | Abstract inference engine |
| [`BaseParser`](luxonis_eval/parsers/base_parser.py) | `parsers/` | Abstract output parser |
| [`BaseMetric`](luxonis_eval/metrics/base_metric.py) | `metrics/` | Abstract evaluation metric |
| [`BaseEvalLoader`](luxonis_eval/loaders/base_loader.py) | `loaders/` | Abstract dataset loader |
| [`BaseVisualizer`](luxonis_eval/visualizers/base_visualizer.py) | `visualizers/` | Abstract result visualizer |

All base classes use the [AutoRegisterMeta](https://github.com/luxonis/luxonis-ml/blob/8b89655497faca6d94e261d49c4d4f96e9078d9b/luxonis_ml/utils/registry.py#L162) metaclass, which means any subclass is **automatically registered** in its component registry and becomes immediately available by name in configuration files — no manual wiring required.

### Evaluation Pipeline: Modular Design

The evaluation loop in `eval_run` is deliberately structured so that each stage of the pipeline depends only on the **abstract interface** of its components, not on any concrete implementation. This is what makes the system fully modular and extensible:

```bash
┌────────────┐     ┌─────────────┐     ┌─────────────┐     ┌───────────┐
│ DataLoader │────▶│    Engine   │────▶│   Parser    │────▶│  Metrics  │
│ (provides  │     │ (runs model │     │ (converts   │     │ (scores   │
│  samples)  │     │  inference) │     │  raw output)│     │  results) │
└────────────┘     └─────────────┘     └─────────────┘     └───────────┘
                                                                   │
                                              ┌────────────┐       │
                                              │ Visualizer │◀──────┘
                                              │ (optional) │
                                              └────────────┘
```

Here is how the pipeline flows:

1. **DataLoader** provides images paired with their ground-truth annotations. Any loader that follows this contract is compatible.

2. **Engine** runs model inference on the image and returns raw outputs. The choice of backend (DepthAI, ONNX Runtime, or a custom one) is transparent to the rest of the pipeline.

3. **Parser** translates the engine's raw outputs into structured predictions that downstream components can consume. Each model architecture can have its own parser.

4. **Metrics** accumulate per-sample results and produce final scores at the end of the run. Multiple metrics can run in parallel, and each one declares which annotation keys it requires.

5. **Visualizer** optionally renders predictions on the input frame for visual inspection.

Because every component is resolved from a registry at runtime based on its **name** in the config, you can mix and match components freely. For example, you can:

- Swap `depthai` for `onnx` in `engine_cfg` without changing anything else
- Add a new metric to `metrics_cfg.metrics` alongside existing ones
- Write a custom parser and reference it by name
- Replace the LDF-based LuxonisLoader dataloader with your own dataset-specific loader

The only constraint is **compatibility**: the parser must produce predictions in the format the metrics expect, and the dataloader must provide annotations with the keys the metrics require (e.g., `"/boundingbox"` for detection metrics, `"/classification"` for accuracy metrics). The `BaseMetric.validate_target_keys()` method catches mismatches early with a clear error message.

## Configuration

Evaluation runs are driven by a YAML configuration file. The configuration is parsed and validated at startup by [`EvalConfig`](luxonis_eval/utils/config.py), which ensures that every referenced component actually exists in its registry and that all required fields are present before the run begins.

A complete configuration file has the following top-level sections:

### Task Name

Defines a human-readable task label used in progress reporting and run output.

```yaml
task_name: InstanceSegmentation
```

### Data Loading & Preprocessing

Specifies which dataloader to use, the dataset it points to, and any preprocessing applied before inference.

```yaml
dataloader_cfg:
  name: LuxonisLoader            # Registered dataloader name
  params:
    dataset_name: coco-2017       # Dataset identifier (required for LuxonisLoader)
    view: [val]                   # Dataset split(s) to use
  preprocessing:
    normalize:
      active: true                # Whether to apply normalization
      params:
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
    color_space: RGB              # RGB | BGR | GRAY
    keep_aspect_ratio: false      # Preserve aspect ratio during resize
```

> [!NOTE]
> When using the `depthai` backend, normalization is usually handled by the model's own preprocessing pipeline. The engine will warn you if normalization is enabled alongside DepthAI. Similarly, DepthAI expects `BGR` color space — a warning is emitted if `RGB` is selected.

### Output Parser

Defines how raw model outputs are converted into structured predictions. Different model architectures produce different output tensor layouts; parsers handle this translation.

```yaml
parser_cfg:
  name: YOLOInstanceSegmentationParser  # Registered parser name
  params:
    conf_thres: 0.25                    # Parser-specific parameters
    mask_thres: 0.25
    iou_thres: 0.45
```

### Evaluation Metrics

A list of metrics to compute. Each metric is independently instantiated, updated per sample, and computed at the end. Throughput is always measured automatically.

```yaml
metrics_cfg:
  metrics:
    - name: BboxMeanAveragePrecision
      params:
        iou_type: bbox
    - name: MaskMeanAveragePrecision
      params:
        iou_type: segm
```

### Visualization (Optional)

Optionally enables live visualization of predictions during the evaluation loop.

```yaml
visualizer_cfg:
  name: InstanceSegmentationVisualizer
  visualize: true                 # Set to false to disable
  params: {}
```

### Inference Engine

Specifies the inference backend and the path to the model file. The config validates that the model file format matches the selected backend (`.tar.xz` → `depthai`, `.onnx` → `onnx`).

```yaml
engine_cfg:
  name: onnx                     # Registered engine name (onnx | depthai)
  model_path: ./models/yolov11n/yolov11n.onnx
  params: {}                      # Engine-specific parameters (e.g., device_ip for RVC4)
```

### Full Example

```yaml
task_name: InstanceSegmentation

dataloader_cfg:
  name: LuxonisLoader
  params:
    dataset_name: coco-2017
    view: [val]
  preprocessing:
    normalize:
      active: false
    color_space: BGR
    keep_aspect_ratio: false

parser_cfg:
  name: YOLOInstanceSegmentationParser
  params:
    conf_thres: 0.25
    mask_thres: 0.25
    iou_thres: 0.45

metrics_cfg:
  metrics:
    - name: BboxMeanAveragePrecision
      params:
        iou_type: bbox
    - name: MaskMeanAveragePrecision
      params:
        iou_type: segm

engine_cfg:
  name: depthai
  model_path: ./models/yolov11n-seg.rvc4.tar.xz
  params:
    device_ip: 192.168.1.100
```

## Extending the Framework

The framework is designed around a principle: **implement a new class inheriting from the respective base class, and the registry takes care of the rest**. Every component type (`BaseEngine`, `BaseEvalLoader`, `BaseParser`, `BaseMetric`, `BaseVisualizer`) uses `AutoRegisterMeta`, which automatically registers new subclasses in the appropriate registry when they are defined. There is no manual registration step — simply subclassing is enough.

### Adding a Custom DataLoader

Every custom loader must inherit from `BaseEvalLoader` and implement four abstract methods:

- **`load_classes()`** – Its return value is assigned to `self.classes`. Must return a `dict[str, int]` mapping class names to integer indices (e.g., `{"cat": 0, "dog": 1}`). The result is validated automatically via `check_loader_classes`.
- **`get_class_mapping()`** – Returns a 3-tuple of `(ldf_class_map, native_class_map, class_index_map)`:
  - **LDF class map** (`dict[int, str]`): How classes are indexed within LuxonisML's data format (LDF), where classes are sorted alphabetically and indices may differ from those used during model training.
  - **Native class map** (`dict[int, str]`): The original class-to-index mapping the model was trained on (e.g., COCO ordering).
  - **Class index map** (`dict[int, int]`): Bridges the two by mapping each LDF index to its corresponding native index, allowing correct alignment of predictions against ground-truth annotations.
  For `LuxonisLoader`-backed datasets, the LDF and native class maps will generally differ and the class index map must explicitly encode the remapping (e.g., `{0: 3, 1: 0, ...}`). For custom datasets that inherit directly from `BaseEvalLoader`, the LDF and native class maps should be identical — both derived from `self.classes` — and the class index map should be an identity mapping (`{0: 0, 1: 1, ...}`).
- **`__getitem__(idx)`** – Returns a `LoaderOutput` tuple for the given index (see note below).
- **`__len__()`** – Returns the number of samples in the dataset.

> [!IMPORTANT]
> **`__getitem__` output format:** The return type must conform to [LoaderOutput](https://github.com/luxonis/luxonis-ml/blob/8b89655497faca6d94e261d49c4d4f96e9078d9b/luxonis_ml/typing.py#L44-L48) from `luxonis_ml.typing`, which is a 2-tuple of `(image, annotations_dict)`:
>
> - **`image`** (`np.ndarray`): A single image as a NumPy array (e.g., shape `(H, W, 3)`).
> - **`annotations_dict`** (`dict[str, np.ndarray]`): A dictionary mapping task-group annotation keys to NumPy arrays. The keys must match what the configured metrics expect — for example, `"/boundingbox"` for detection, `"/classification"` for classification, `"/segmentation"` for segmentation, etc.
>
> Every subclass's `__getitem__` is automatically wrapped by the `@validate_loader_output` decorator, which calls `check_loader_output` at runtime to verify the tuple structure and types. If the output doesn't match the expected format, a `TypeError` is raised with a descriptive message indicating the loader class and sample index.

### Adding a Custom Engine

Subclass [`BaseEngine`](luxonis_eval/engines/base_engine.py) and implement the six abstract methods:

- **`setup()`** – Initializes backend resources (e.g., loading a runtime, connecting to a device, building a pipeline).
- **`get_input_shape()`** – Returns the model's expected input dimensions as a `(width, height)` tuple. Used to resize dataloader images before inference.
- **`get_platform_name()`** – Returns a human-readable platform identifier (e.g., `"RVC2"`, `"RVC4"`). Displayed in the evaluation report.
- **`infer_once(img)`** – Runs inference on a single preprocessed image (`np.ndarray`) and returns the raw backend output. The return type is backend-specific — parsers handle the translation.
- **`vis_frame()`** – Returns a copy of the input image suitable for visualization overlays.
- **`teardown()`** – Releases all backend resources (sessions, devices, pipelines). Called automatically after the evaluation loop finishes.

### Adding a Custom Parser

Subclass [`BaseParser`](luxonis_eval/parsers/base_parser.py) and implement the single abstract method:

- **`parse(raw_output, **kwargs)`** – Converts the raw backend output into a structured prediction. The `raw_output` type depends on the engine being used (`dai.NNData` for DepthAI, `list[np.ndarray]` for ONNX Runtime, or whatever a custom engine returns). Additional keyword arguments are forwarded from `parser_cfg.params` in the config and the evaluation loop.

The parser bridges the gap between a specific model architecture's raw tensor layout and the standardized format that downstream metrics expect. The standardized format used for the built-in supported parsers are as follows:

- [**ClassificationParser**](luxonis_eval/parsers/classification.py) –> [depthai_nodes.Classifications](https://github.com/luxonis/depthai-nodes/tree/main/depthai_nodes/message#classifications)
- [**YOLODetectionParser**](luxonis_eval/parsers/detection.py) –> [dai.ImgDetections](https://docs.luxonis.com/software-v3/depthai/api/cpp/#classdai_1_1ImgDetections)
- [**YOLOInstanceSegmentationParser**](luxonis_eval/parsers/instance_seg.py) –> [dai.ImgDetections](https://docs.luxonis.com/software-v3/depthai/api/cpp/#classdai_1_1ImgDetections)
- [**YOLOKeypointDetectionParser**](luxonis_eval/parsers/keypoint_detection.py) –> [dai.ImgDetections](https://docs.luxonis.com/software-v3/depthai/api/cpp/#classdai_1_1ImgDetections)
- [**SemanticSegmentationParser**](luxonis_eval/parsers/semantic_seg.py) –> [depthai_nodes.SegmentationMask](https://github.com/luxonis/depthai-nodes/tree/main/depthai_nodes/message#segmentationmask)

> [!IMPORTANT]
> The parser must produce predictions in the format that the configured metrics expect. For example, for the detection metric `BboxMeanAveragePrecision`, the parser should return a message of type `dai.ImgDetections` which is what the metric expects at its input. Mismatches between annotation keys and metric requirements are caught early by `BaseMetric.validate_target_keys()`.

### Adding a Custom Metric

Subclass `BaseMetric` and implement the four abstract methods:

- **`metric_keys()`** – Declares which annotation keys this metric requires — the framework validates their presence in the ground-truth data automatically.
- **`_reset_impl()`** – Resets internal state (e.g., counters, accumulators).
- **`_update_impl(predictions, target, **kwargs)`** – Updates internal state with predictions and ground-truth for a sample.
- **`_compute_impl()`** – Computes and returns the final metric value.

> [!IMPORTANT]
> Similar to the parser, the metric must be able to process the output of the corresponding parser. If a parser returns a `dai.ImgDetections` message, then the metric must be able to process that message.

### General Pattern

All extensions follow the same three-step pattern:

1. **Subclass** the appropriate base class
2. **Implement** the abstract methods
3. **Reference by name** in your YAML config

No imports, no registration calls, no factory functions — the metaclass handles everything. As long as your module is imported (which happens automatically for files inside the `luxonis_eval/` package), the class is available.

## License

This project is licensed under the [Apache License 2.0](LICENSE).
