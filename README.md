# LuxonisEval

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

<a name="overview"></a>

## 🌟 Overview

`LuxonisEval` is a modular evaluation framework for benchmarking neural network models across multiple inference engines. It supports on-device inference on Luxonis devices (`RVC2` and `RVC4`) through `DepthAI`, as well as host-side inference through `ONNX Runtime`, while reporting both quality metrics and throughput or latency performance.

The framework follows a registry-based architecture: each pluggable component (engines, dataloaders, parsers, metrics, and visualizers) registers itself automatically. This lets you swap, extend, or add parts of the evaluation pipeline without modifying the core evaluation loop. In practice, adding a new component usually means subclassing the appropriate base class and referencing it by name in the configuration.

### ✨ Key Features

- **Multiple Inference Engines**
  - [**DepthAI Engine**](luxonis_eval/engines/depthai_engine.py) - Run models exported as [NNArchive](https://docs.luxonis.com/software-v3/ai-inference/nn-archive) files on Luxonis devices via [DepthAI](https://docs.luxonis.com/software-v3/depthai/)
  - [**ONNX Engine**](luxonis_eval/engines/onnx_engine.py) - Run models on CPU or GPU using `ONNX Runtime`
- **Dataset Loading**
  - [**LuxonisLoader**](https://github.com/luxonis/luxonis-ml/tree/main/luxonis_ml/data/loaders#luxonisml-loader) - Load datasets stored in Luxonis Data Format (`LDF`)
  - [**BaseEvalLoader**](luxonis_eval/loaders/base_loader.py) - Base class for custom dataloaders
- **Current Evaluation Coverage** - The built-in parsers and metrics currently cover classification, bounding box detection, semantic segmentation, instance segmentation, and keypoint evaluation
- **NNArchive-Aware Configuration** - Parser metadata and preprocessing hints can be resolved from NNArchive models, including archive-driven overrides when desired
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
  - [🧠 Evaluators](#evaluators)
  - [🎨 Visualizers](#visualizers)
  - [⚡ Inference Engine](#inference-engine)
  - [📄 Full Example](#full-example)
  - [📏 Metrics](#metrics)
  - [🏃 Commands](#commands)
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

# Use the ONNX engine
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
from luxonis_eval.config import EvalConfig

eval_cfg = EvalConfig.get_config(cfg="path/to/config.yaml")
results = eval_run(eval_cfg)
```

If you need explicit lifecycle control, repeated `evaluate()` calls after one
setup, or direct access to runtime state, use `LuxonisEval` directly:

```python
from luxonis_eval import LuxonisEval
from luxonis_eval.config import EvalConfig

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
├── config/           # Configuration schema and exports
├── core/             # Evaluation lifecycle orchestration
├── engines/          # Inference engines
├── loaders/          # Dataset loaders
├── metrics/          # Evaluation metrics
├── parsers/          # Model output parsers
├── utils/            # Shared low-level helpers
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

The evaluation loop in `LuxonisEval.evaluate()` is structured around abstract component interfaces rather than concrete implementations. That design keeps the pipeline modular and makes engine-specific or model-specific components easy to replace.

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
2. **Engine** runs inference and returns raw engine outputs.
3. **Parser** converts raw engine outputs into a structured prediction format.
4. **Metrics** accumulate per-sample results and compute final scores.
5. **Visualizer** optionally renders predictions for inspection.

Because each component is resolved from a registry at runtime, you can mix and match implementations freely. For example, you can:

- swap `depthai` for `onnx` in `engine` without changing the rest of the config
- add another metric under `metrics.metrics`
- introduce a custom parser and reference it by name
- replace `LuxonisLoader` with a dataset-specific custom loader

The main constraint is compatibility: the parser must produce predictions in the format the configured metrics expect, and the dataloader must provide the annotation keys those metrics require. `LuxonisEval.setup()` runs a one-sample pipeline sanity check across loader, engine, parser, and metrics so incompatible configurations fail before evaluation starts.

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

Evaluation runs are driven by a YAML configuration file. [`EvalConfig`](luxonis_eval/config/config.py) parses and validates the configuration at startup, ensuring that referenced components exist and that required fields are present before evaluation begins.

A typical configuration file only needs the `pipeline` block. `version` is filled from the package version, and `runtime` uses its default empty value when omitted.

```yaml
pipeline:
  loader: ...
  engine: ...
  evaluators:
    - ...
```

### 📦 Data Loading And Preprocessing

This section defines which dataloader to use, which dataset it points to, and which preprocessing steps are applied before inference.

```yaml
pipeline:
  loader:
    name: LuxonisLoader           # Registered dataloader name
    params:
      dataset_name: coco-2017     # Dataset identifier
      view: [val]                 # Dataset split(s) to use
    preprocessing:
      normalize:
        active: true              # Whether to apply normalization
        params:
          mean: [0.485, 0.456, 0.406]
          std: [0.229, 0.224, 0.225]
      color_space: RGB            # RGB | BGR | GRAY
      keep_aspect_ratio: true     # Preserve aspect ratio during resize
```

`preprocessing` is resolved before evaluation starts. For ordinary configs, the loader values come directly from YAML. When the model path points to an NNArchive, LuxonisEval can also derive preprocessing hints from the archive metadata.

```yaml
runtime:
  nn_archive_params_override: true
pipeline:
  loader:
    preprocessing:
      keep_aspect_ratio: true
      normalize:
        active: false
      color_space: RGB
```

When `runtime.nn_archive_params_override` is `true`, NNArchive metadata takes precedence for:

- `loader.preprocessing.normalize`
- `loader.preprocessing.color_space`
- evaluator parser selection and parser params
- evaluator `outputs`

When it is `false`, explicit YAML values stay primary and archive metadata is only used as a fallback.

> [!IMPORTANT]
> `keep_aspect_ratio` is not inferred from NNArchive metadata. Set it explicitly when your preprocessing depends on preserving aspect ratio or letterboxing.

> [!NOTE]
> For the `depthai` engine, host-side normalization is skipped because preprocessing is expected to run on-device through the NNArchive pipeline. For the `onnx` engine, resolved normalization stays on the host side.

> [!IMPORTANT]
> `LuxonisLoader` evaluation is currently single-evaluator and single-dataset-task only. `pipeline.evaluators[*].task_name` selects the Luxonis dataset task namespace to evaluate. It is not a framework-level task enum or abstraction. For datasets that use the default empty Luxonis task, set `task_name: ""`.

### 🧠 Evaluators

Each pipeline evaluator binds together one dataset task selection, one parser, a set of metrics, and optional visualizers for one quality-evaluation unit.

```yaml
pipeline:
  evaluators:
    - task_name: instance_segmentation
      parser:
        name: YOLOExtendedParser
        params:
          subtype: yolov8
          n_classes: 80
          conf_threshold: 0.25
          iou_threshold: 0.7
          mask_conf: 0.25
      metrics:
        - name: BboxMeanAveragePrecision
          params:
            iou_type: bbox
        - name: MaskMeanAveragePrecision
          params:
            iou_type: segm
      visualizers: []
```

- `task_name` selects the Luxonis dataset task evaluated by this entry. If not set we try to infer it from the dataset metadata.
- `name` is optional and defaults to `task_name` or a stable fallback when `task_name` is empty.
- `outputs` is optional in the current single-evaluator implementation; when omitted, the evaluator consumes all engine outputs.
- Only zero or one evaluator is currently supported at runtime. Multiple evaluators are rejected with a clear not-yet-implemented error.

Compatibility is driven by data shape, not by a separate task abstraction:

- the loader must expose the annotation keys required by the configured metrics
- the parser must produce outputs that the configured metrics can consume

### 🎨 Visualizers

Visualizers are evaluator-local and plural. The repository currently ships
only the `BaseVisualizer` interface, so visualizer entries are only useful
when you provide and import a custom implementation.

### ⚡ Inference Engine

The engine section selects the inference engine and points to the model file. Configuration validation ensures that the model format matches the selected engine (`.tar.xz` NNArchive for `depthai` or `onnx`, `.onnx` for `onnx`).

```yaml
pipeline:
  engine:
    name: onnx                    # Registered engine name: onnx | depthai
    model_path: ./models/yolov11n/yolov11n.onnx
    params: {}                    # Engine-specific parameters, for example device_ip for RVC4
```

> [!NOTE]
> The CLI override flag is named `--backend` for convenience, but it simply overrides `pipeline.engine.name`.

### 📄 Full Example

```yaml
runtime:
  nn_archive_params_override: false

pipeline:
  loader:
    name: LuxonisLoader
    preprocessing:
      keep_aspect_ratio: true
      color_space: RGB
      normalize:
        active: true
        params:
          mean: [0.0, 0.0, 0.0]
          std: [1.0, 1.0, 1.0]
    params:
      dataset_name: coco-2017
      view: [val]

  engine:
    name: onnx
    model_path: examples/quickstart_inst_seg/models/yolov8n-seg.onnx
    params:
      providers: [CPUExecutionProvider]

  evaluators:
    - task_name: instance_segmentation
      parser:
        name: YOLOExtendedParser
        params:
          subtype: yolov8
          n_classes: 80
          conf_threshold: 0.25
          iou_threshold: 0.7
          mask_conf: 0.25
      metrics:
        - name: BboxMeanAveragePrecision
          params:
            iou_type: bbox
        - name: MaskMeanAveragePrecision
          params:
            iou_type: segm
      visualizers: []
```

### 📏 Metrics

Quality metrics are configured per evaluator under `pipeline.evaluators[*].metrics`.

| Metric | Typical use | Required target keys |
| --- | --- | --- |
| [`TopKAccuracy`](luxonis_eval/metrics/topk_accuracy.py) | Classification | `["/classification"]` |
| [`BboxMeanAveragePrecision`](luxonis_eval/metrics/bbox_map.py) | Bounding box detection | `["/boundingbox"]` |
| [`MaskMeanAveragePrecision`](luxonis_eval/metrics/mask_map.py) | Instance segmentation | `["/boundingbox", "/instance_segmentation"]` |
| [`KeypointMeanAveragePrecision`](luxonis_eval/metrics/keypoint_map.py) | Keypoint evaluation | `["/boundingbox", "/keypoints"]` |
| [`MIoU`](luxonis_eval/metrics/mIoU.py) | Semantic segmentation | `["/segmentation"]` |
| [`DiceCoefficient`](luxonis_eval/metrics/dice_coef.py) | Semantic segmentation | `["/segmentation"]` |
| [`F1Score`](luxonis_eval/metrics/f1_score.py) | Semantic segmentation | `["/segmentation"]` |
| [`JaccardIndex`](luxonis_eval/metrics/jaccard_index.py) | Semantic segmentation | `["/segmentation"]` |

Metrics consume parser outputs directly. Each metric validates that the parser returned the message type it expects, for example classifications, segmentation masks, detections, or detections with attached instance-mask metadata.

`ThroughputMetric` is not configured manually in the evaluator list. It is always collected internally and reported alongside the quality metrics in the final `EvaluationResult`.

### 🏃 Commands

`luxonis_eval eval --config ...` runs the configured quality pipeline in this phase.

`luxonis_eval quality --config ...` is a quality-only alias with the same override flags as `eval`.

Benchmark configuration can be present in the YAML for future compatibility, but benchmark execution is intentionally not implemented yet.

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

For `LuxonisLoader`-backed datasets, the LDF and native class maps may differ when the model was trained with a different class order than the dataset metadata, so the class index map must encode that remapping explicitly. If no native class mapping is provided for an unknown `LuxonisLoader` dataset, LuxonisEval falls back to the dataset's LDF class order and uses an identity class index map after issuing a warning. For custom datasets that inherit directly from `BaseEvalLoader`, the two class maps are usually identical and the class index map is typically an identity mapping.

> [!IMPORTANT]
> `__getitem__` must return [LoaderOutput](https://github.com/luxonis/luxonis-ml/blob/8b89655497faca6d94e261d49c4d4f96e9078d9b/luxonis_ml/typing.py#L44-L48) from `luxonis_ml.typing`, which is a tuple of `(image, annotations_dict)`.
>
> - **`image`** (`np.ndarray`) is a single image, for example with shape `(H, W, 3)`.
> - **`annotations_dict`** (`dict[str, np.ndarray]`) maps task-group annotation keys to arrays, such as `"/boundingbox"`, `"/classification"`, or `"/segmentation"`.
> - **`model_spec`** (`ModelSpec`) is passed to every loader constructor by LuxonisEval, so custom loaders can use the engine-resolved `width` and `height` during initialization or preprocessing setup.
>
> Every subclass implementation of `__getitem__` is wrapped by `@validate_loader_output`, which calls `check_loader_output` at runtime and raises a descriptive `TypeError` if the output format is invalid.
>
> The loader must also provide a schema-stable `annotations_dict`: every sample must expose the same annotation keys. If a metric requires a key, that key must be present for every sample.

### 🔌 Adding a Custom Engine

Subclass [`BaseEngine`](luxonis_eval/engines/base_engine.py) and implement the four abstract methods:

- **`setup()`** - Initialize engine resources such as runtimes, sessions, or device connections, then return a `ModelSpec(width, height)` for the loaded model. Keep this idempotent so repeated calls are safe.
- **`infer_once(img)`** - Run inference on a single preprocessed image and return the raw engine output
- **`vis_frame()`** - Return a copy of the input image suitable for visualization overlays
- **`close()`** - Release engine resources after evaluation finishes

The framework consumes the returned `ModelSpec` to configure loader preprocessing and metric contexts.

### 🧠 Adding a Custom Parser

Subclass [`BaseParser`](luxonis_eval/parsers/base_parser.py) and implement the single abstract method:

- **`parse(output)`** - Convert raw engine output into a structured prediction format

Parser configuration belongs in the parser itself, and LuxonisEval provides the remaining runtime information during setup.

The parser bridges the gap between model-specific tensor layouts and the standardized message types that downstream metrics expect. The built-in parsers produce the following output types:

- [**ClassificationParser**](luxonis_eval/parsers/classification.py) -> [depthai_nodes.Classifications](https://github.com/luxonis/depthai-nodes/tree/main/depthai_nodes/message#classifications)
- [**YOLOExtendedParser**](luxonis_eval/parsers/yolo.py) -> [dai.ImgDetections](https://docs.luxonis.com/software-v3/depthai/api/cpp/#classdai_1_1ImgDetections)
- [**SegmentationParser**](luxonis_eval/parsers/segmentation.py) -> [dai.SegmentationMask](https://github.com/luxonis/depthai-nodes/tree/main/depthai_nodes/message#segmentationmask)

> [!IMPORTANT]
> The parser must produce outputs that the configured metrics can consume. For example, if a metric expects `dai.ImgDetections`, the parser must return that message type.

### 📐 Adding a Custom Metric

Subclass [`BaseMetric`](luxonis_eval/metrics/base_metric.py) and implement the four abstract methods:

- **`required_target_keys()`** - Declare which annotation keys the metric requires
- **`reset()`** - Reset internal state such as counters or accumulators
- **`update(predictions, target)`** - Update the metric state for one sample
- **`compute()`** - Return the final metric values

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
