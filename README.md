# LuxonisEval

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

<a name="overview"></a>

## 🌟 Overview

`LuxonisEval` is a modular evaluation framework for benchmarking neural network models across multiple inference backends. It supports inference on Luxonis devices (`RVC2` and `RVC4`) through `DepthAI`, as well as host-side inference through `ONNX Runtime`, while reporting both quality metrics and throughput or latency performance.

The framework follows a registry-based architecture: each pluggable component (engines, dataloaders, parsers, metrics, and visualizers) registers itself automatically. This lets you swap, extend, or add parts of the evaluation pipeline without modifying the core evaluation loop. In practice, adding a new component usually means subclassing the appropriate base class and referencing it by name in the configuration.

### ✨ Key Features

- **Multiple Inference Backends**
  - [**DepthAI Engine**](luxonis_eval/engines/depthai_engine.py) - Run models exported as [NNArchive](https://docs.luxonis.com/software-v3/ai-inference/nn-archive) files on Luxonis devices via [DepthAI](https://docs.luxonis.com/software-v3/depthai/)
  - [**ONNX Engine**](luxonis_eval/engines/onnx_engine.py) - Run models on CPU or GPU using `ONNX Runtime`
- **Dataset Loading**
  - [**LuxonisLoader**](https://github.com/luxonis/luxonis-ml/tree/main/luxonis_ml/data/loaders#luxonisml-loader) - Load datasets stored in Luxonis Data Format (`LDF`)
  - [**BaseEvalLoader**](luxonis_eval/loaders/base_loader.py) - Base class for custom dataloaders
- **Supported Tasks**
  - `Classification` - Image classification
  - `Detection` - Bounding box detection
  - `SemanticSegmentation` - Per-pixel class labeling
  - `InstanceSegmentation` - Per-instance masks with detection
  - `KeypointDetection` - Body or object keypoint localization
- **Built-In Metrics**
  - [`TopKAccuracy`](luxonis_eval/metrics/topk_accuracy.py) - Top-1 and Top-5 accuracy for classification
  - [`BboxMeanAveragePrecision`](luxonis_eval/metrics/bbox_map.py) - COCO-style mAP for bounding box detection
  - [`MaskMeanAveragePrecision`](luxonis_eval/metrics/mask_map.py) - COCO-style mAP for instance segmentation
  - [`KeypointMeanAveragePrecision`](luxonis_eval/metrics/keypoint_map.py) - OKS-based mAP for keypoint detection
  - [`MIoU`](luxonis_eval/metrics/mIoU.py) - Mean Intersection over Union for semantic segmentation
  - [`DiceCoefficient`](luxonis_eval/metrics/dice_coef.py) - Dice score for semantic segmentation
  - [`ThroughputMetric`](luxonis_eval/metrics/throughput.py) - End-to-end throughput and latency reporting
- **Extensible Architecture** - Registry-based design powered by [`AutoRegisterMeta`](luxonis_eval/registry.py), making it straightforward to add custom engines, parsers, metrics, loaders, and visualizers

<a name="quick-start"></a>

## 🚀 Quick Start

Get started with `LuxonisEval` in a few steps:

1. **Install the project from source**

   ```bash
   pip install .
   ```

2. **Prepare the example model and dataset (requires the `fiftyone` package)**

   ```bash
   pip install fiftyone
   bash examples/quickstart_inst_seg/setup_example.sh
   ```

3. **Run the evaluation**

   ```bash
   luxonis_eval eval --config configs/yolov8n_inst_seg_config.yaml
   ```

This quickstart runs instance segmentation evaluation with `ONNX Runtime` on CPU and does not require Luxonis hardware. For a fuller walkthrough, see [examples/quickstart_inst_seg/README.md](examples/quickstart_inst_seg/README.md).

## Table Of Contents

- [🌟 Overview](#overview)
  - [✨ Key Features](#key-features)
- [🚀 Quick Start](#quick-start)
- [🛠️ Installation](#installation)
- [📝 Usage](#usage)
  - [💻 CLI](#cli)
  - [🐍 Python API](#python-api)
- [🏗️ Architecture](#architecture)
  - [🧩 Key Base Classes](#key-base-classes)
  - [🔄 Evaluation Pipeline](#evaluation-pipeline)
  - [📊 Throughput Metric Semantics](#throughput-metric-semantics)
- [⚙️ Configuration](#configuration)
  - [📦 Data Loading And Preprocessing](#data-loading-and-preprocessing)
  - [🧠 Output Parser](#output-parser)
  - [📏 Evaluation Metrics](#evaluation-metrics)
  - [🎨 Visualization](#visualization)
  - [⚡ Inference Engine](#inference-engine)
  - [📄 Full Example](#full-example)
- [🧱 Extending the Framework](#extending-the-framework)
  - [📥 Adding a Custom DataLoader](#adding-a-custom-dataloader)
  - [🔌 Adding a Custom Engine](#adding-a-custom-engine)
  - [🧠 Adding a Custom Parser](#adding-a-custom-parser)
  - [📐 Adding a Custom Metric](#adding-a-custom-metric)
  - [🪜 General Pattern](#general-pattern)
- [📄 License](#license)

<a name="installation"></a>

## 🛠️ Installation

`LuxonisEval` requires **Python 3.10** or higher. We recommend using a virtual environment to keep dependencies isolated.

**Install from source**:

```bash
pip install .
```

This installs the `luxonis_eval` CLI in your environment.

**Developer install**:

```bash
pip install -e ".[dev]"
```

<a name="usage"></a>

## 📝 Usage

You can use `LuxonisEval` either from the command line or through the Python API. The CLI is the primary entry point for running evaluations from configuration files.

<a name="cli"></a>

### 💻 CLI

The CLI currently exposes the `eval` command:

```bash
luxonis_eval eval --help
```

Example invocations:

```bash
# Run evaluation with a config file
luxonis_eval eval --config path/to/config.yaml

# Run with CLI overrides
luxonis_eval eval \
    --config path/to/config.yaml \
    --dataset-name coco \
    --model-path path/to/model.tar.xz \
    --backend depthai

# Use the ONNX backend
luxonis_eval eval \
    --config path/to/config.yaml \
    --dataset-name coco \
    --model-path path/to/model.onnx \
    --backend onnx

# Specify device IP for RVC4
luxonis_eval eval \
    --config path/to/config.yaml \
    --device-ip 192.168.1.100
```

<a name="python-api"></a>

### 🐍 Python API

For one-shot programmatic usage, call `eval_run`:

```python
from luxonis_eval.__main__ import eval_run
from luxonis_eval.utils.config import EvalConfig

eval_cfg = EvalConfig.get_config(cfg="path/to/config.yaml")
results = eval_run(eval_cfg)
```

If you need explicit lifecycle control, repeated `evaluate()` calls after one
setup, or direct access to runtime state, use `LuxonisEval` directly:

```python
from luxonis_eval import LuxonisEval
from luxonis_eval.utils.config import EvalConfig

eval_cfg = EvalConfig.get_config(cfg="path/to/config.yaml")
evaluator = LuxonisEval(eval_cfg)
evaluator.setup()
results = evaluator.evaluate()
evaluator.close()
```

<a name="architecture"></a>

## 🏗️ Architecture

The repository is organized around a small set of core component types:

```bash
luxonis_eval/
├── engines/          # Inference backends
├── loaders/          # Dataset loaders
├── metrics/          # Evaluation metrics
├── parsers/          # Model output parsers
├── utils/            # Configuration and helper functions
├── visualizers/      # Result visualization
└── metadata/         # Class mapping files
```

### 🧩 Key Base Classes

| Base Class                                                      | Location       | Purpose                    |
| --------------------------------------------------------------- | -------------- | -------------------------- |
| [`BaseEngine`](luxonis_eval/engines/base_engine.py)             | `engines/`     | Abstract inference engine  |
| [`BaseParser`](luxonis_eval/parsers/base_parser.py)             | `parsers/`     | Abstract output parser     |
| [`BaseMetric`](luxonis_eval/metrics/base_metric.py)             | `metrics/`     | Abstract evaluation metric |
| [`BaseEvalLoader`](luxonis_eval/loaders/base_loader.py)         | `loaders/`     | Abstract dataset loader    |
| [`BaseVisualizer`](luxonis_eval/visualizers/base_visualizer.py) | `visualizers/` | Abstract result visualizer |

All base classes use the [AutoRegisterMeta](https://github.com/luxonis/luxonis-ml/blob/8b89655497faca6d94e261d49c4d4f96e9078d9b/luxonis_ml/utils/registry.py#L162) metaclass. Any subclass is registered automatically and becomes available by name in configuration files, with no manual wiring required.

### 🔄 Evaluation Pipeline

The evaluation loop in `LuxonisEval.evaluate()` is structured around abstract component interfaces rather than concrete implementations. That design keeps the pipeline modular and makes backend or task-specific components easy to replace.

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

The pipeline works as follows:

1. **DataLoader** provides images together with ground-truth annotations.
2. **Engine** runs inference and returns raw backend outputs.
3. **Parser** converts raw outputs into a structured prediction format.
4. **Metrics** accumulate per-sample results and compute final scores.
5. **Visualizer** optionally renders predictions for inspection.

Because each component is resolved from a registry at runtime, you can mix and match implementations freely. For example, you can:

- swap `depthai` for `onnx` in `engine` without changing the rest of the config
- add another metric under `metrics.metrics`
- introduce a custom parser and reference it by name
- replace `LuxonisLoader` with a dataset-specific custom loader

The main constraint is compatibility: the parser must produce predictions in the format the configured metrics expect, and the dataloader must provide the annotation keys those metrics require. `BaseMetric.validate_target_keys()` catches mismatches early and raises a clear error message.

<a name="throughput-metric-semantics"></a>

### 📊 Throughput Metric Semantics

`ThroughputMetric` measures end-to-end pipeline timing rather than isolated model-only benchmarks. The reported rows mean:

> [!WARNING]
> Throughput values are end-to-end pipeline measurements and not isolated model-only benchmarks. Lower numbers than `modelconverter` benchmark results are expected.

- **Throughput** - Samples processed per second across the full evaluation pipeline
- **End-to-end Latency** - Average wall-clock time per sample for the whole run
- **Inference** - Time spent inside the inference engine
- **Parsing** - Time spent converting raw model outputs into predictions
- **Metric Update** - Time spent updating metrics for each sample
- **Metric Compute** - Time spent in the final metric aggregation after the sample loop
- **Pipeline Overhead** - Remaining time not covered by the rows above; this typically includes dataloader iteration, image decode, preprocessing such as resize or normalization, annotation reconstruction, visualization, progress bar updates, and general loop bookkeeping

Rule of thumb: `End-to-end Latency ≈ Inference + Parsing + Metric Update + Metric Compute + Pipeline Overhead`

<a name="configuration"></a>

## ⚙️ Configuration

Evaluation runs are driven by a YAML configuration file. [`EvalConfig`](luxonis_eval/utils/config.py) parses and validates the configuration at startup, ensuring that referenced components exist and that required fields are present before evaluation begins.

A complete configuration file is typically organized into the sections below.

### 📦 Data Loading And Preprocessing

This section defines which dataloader to use, which dataset it points to, and which preprocessing steps are applied before inference.

```yaml
loader:
  name: LuxonisLoader             # Registered dataloader name
  params:
    dataset_name: coco-2017       # Dataset identifier
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
> When using the `depthai` backend, normalization is usually handled by the model's own preprocessing pipeline. The engine will warn you if normalization is enabled together with `DepthAI`. `DepthAI` also expects `BGR` color space, so a warning is emitted if `RGB` is selected.

### 🧠 Output Parser

The parser converts raw model outputs into structured predictions. Different model architectures expose different tensor layouts, so the parser is responsible for translating backend-specific outputs into a format the metrics can consume.

```yaml
parser:
  name: YOLOInstanceSegmentationParser
  params:
    conf_thres: 0.25
    mask_thres: 0.25
    iou_thres: 0.45
```

### 📏 Evaluation Metrics

Metrics are instantiated independently, updated for each sample, and computed at the end of the run. Throughput reporting is added automatically.

```yaml
metrics:
  metrics:
    - name: BboxMeanAveragePrecision
      params:
        iou_type: bbox
    - name: MaskMeanAveragePrecision
      params:
        iou_type: segm
```

### 🎨 Visualization

Visualization is optional and can be enabled when you want to inspect predictions during the evaluation loop.

```yaml
visualizer:
  name: InstanceSegmentationVisualizer
  visualize: true
  params: {}
```

### ⚡ Inference Engine

The engine section selects the backend and points to the model file. Configuration validation ensures that the model format matches the backend (`.tar.xz` for `depthai`, `.onnx` for `onnx`).

```yaml
engine:
  name: onnx                      # Registered engine name: onnx | depthai
  model_path: ./models/yolov11n/yolov11n.onnx
  params: {}                      # Engine-specific parameters, for example device_ip for RVC4
```

### 📄 Full Example

```yaml
loader:
  name: LuxonisLoader
  params:
    dataset_name: coco-2017
    view: [val]
  preprocessing:
    normalize:
      active: false
    color_space: BGR
    keep_aspect_ratio: false

parser:
  name: YOLOInstanceSegmentationParser
  params:
    conf_thres: 0.25
    mask_thres: 0.25
    iou_thres: 0.45

metrics:
  metrics:
    - name: BboxMeanAveragePrecision
      params:
        iou_type: bbox
    - name: MaskMeanAveragePrecision
      params:
        iou_type: segm

engine:
  name: depthai
  model_path: ./models/yolov11n-seg.rvc4.tar.xz
  params:
    device_ip: 192.168.1.100
```

<a name="extending-the-framework"></a>

## 🧱 Extending the Framework

`LuxonisEval` is designed around a simple rule: implement a new class that inherits from the appropriate base class, and the registry handles the rest. Every component type (`BaseEngine`, `BaseEvalLoader`, `BaseParser`, `BaseMetric`, `BaseVisualizer`) uses `AutoRegisterMeta`, so subclassing is enough to make a component available once its module is imported.

### 📥 Adding a Custom DataLoader

Every custom loader must inherit from `BaseEvalLoader` and implement four abstract methods:

- **`load_classes()`** - Returns a `dict[str, int]` mapping class names to integer indices. The result is assigned to `self.classes` and validated automatically.
- **`get_class_mapping()`** - Returns a tuple of `(ldf_class_map, native_class_map, class_index_map)`:
  - **LDF class map** (`dict[int, str]`): class ordering used inside Luxonis Data Format
  - **Native class map** (`dict[int, str]`): original class ordering used during training
  - **Class index map** (`dict[int, int]`): mapping from LDF indices to native indices
- **`__getitem__(idx)`** - Returns a `LoaderOutput` tuple for the requested sample
- **`__len__()`** - Returns the number of samples in the dataset

For `LuxonisLoader`-backed datasets, the LDF and native class maps often differ, so the class index map must encode the remapping explicitly. For custom datasets that inherit directly from `BaseEvalLoader`, the two class maps are usually identical and the class index map is typically an identity mapping.

> [!IMPORTANT]
> `__getitem__` must return [LoaderOutput](https://github.com/luxonis/luxonis-ml/blob/8b89655497faca6d94e261d49c4d4f96e9078d9b/luxonis_ml/typing.py#L44-L48) from `luxonis_ml.typing`, which is a tuple of `(image, annotations_dict)`.
>
> - **`image`** (`np.ndarray`) is a single image, for example with shape `(H, W, 3)`.
> - **`annotations_dict`** (`dict[str, np.ndarray]`) maps task-group annotation keys to arrays, such as `"/boundingbox"`, `"/classification"`, or `"/segmentation"`.
>
> Every subclass implementation of `__getitem__` is wrapped by `@validate_loader_output`, which calls `check_loader_output` at runtime and raises a descriptive `TypeError` if the output format is invalid.

### 🔌 Adding a Custom Engine

Subclass [`BaseEngine`](luxonis_eval/engines/base_engine.py) and implement the six abstract methods:

- **`setup()`** - Initialize backend resources such as runtimes, sessions, or device connections
- **`get_input_shape()`** - Return the model input size as a `(width, height)` tuple
- **`get_platform_name()`** - Return a human-readable platform name such as `"RVC2"` or `"RVC4"`
- **`infer_once(img)`** - Run inference on a single preprocessed image and return the raw backend output
- **`vis_frame()`** - Return a copy of the input image suitable for visualization overlays
- **`close()`** - Release backend resources after evaluation finishes

### 🧠 Adding a Custom Parser

Subclass [`BaseParser`](luxonis_eval/parsers/base_parser.py) and implement the single abstract method:

- **`parse(raw_output, **kwargs)`** - Convert raw backend output into a structured prediction format

The parser bridges the gap between model-specific tensor layouts and the standardized message types that downstream metrics expect. The built-in parsers produce the following output types:

- [**ClassificationParser**](luxonis_eval/parsers/classification.py) -> [depthai_nodes.Classifications](https://github.com/luxonis/depthai-nodes/tree/main/depthai_nodes/message#classifications)
- [**YOLODetectionParser**](luxonis_eval/parsers/detection.py) -> [dai.ImgDetections](https://docs.luxonis.com/software-v3/depthai/api/cpp/#classdai_1_1ImgDetections)
- [**YOLOInstanceSegmentationParser**](luxonis_eval/parsers/instance_seg.py) -> [dai.ImgDetections](https://docs.luxonis.com/software-v3/depthai/api/cpp/#classdai_1_1ImgDetections)
- [**YOLOKeypointDetectionParser**](luxonis_eval/parsers/keypoint_detection.py) -> [dai.ImgDetections](https://docs.luxonis.com/software-v3/depthai/api/cpp/#classdai_1_1ImgDetections)
- [**SemanticSegmentationParser**](luxonis_eval/parsers/semantic_seg.py) -> [depthai_nodes.SegmentationMask](https://github.com/luxonis/depthai-nodes/tree/main/depthai_nodes/message#segmentationmask)

> [!IMPORTANT]
> The parser must produce outputs that the configured metrics can consume. For example, if a metric expects `dai.ImgDetections`, the parser must return that message type.

### 📐 Adding a Custom Metric

Subclass [`BaseMetric`](luxonis_eval/metrics/base_metric.py) and implement the four abstract methods:

- **`metric_keys()`** - Declare which annotation keys the metric requires
- **`_reset_impl()`** - Reset internal state such as counters or accumulators
- **`_update_impl(predictions, target, **kwargs)`** - Update the metric state for one sample
- **`_compute_impl()`** - Return the final metric value

> [!IMPORTANT]
> Metrics must be compatible with the outputs generated by the configured parser. If the parser returns `dai.ImgDetections`, the metric must know how to process that object.

### 🪜 General Pattern

All extensions follow the same three-step workflow:

1. **Subclass** the appropriate base class
2. **Implement** the required abstract methods
3. **Reference the component by name** in the YAML config

No manual registration, factory wiring, or extra boilerplate is required. As long as the module is imported, the metaclass makes the class available.

<a name="license"></a>

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).
