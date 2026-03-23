# Quickstart Example: YOLOv8 Instance Segmentation with ONNX Runtime

## Overview

This example shows a minimal end-to-end `luxonis-eval` workflow that runs entirely on the host and does not require a Luxonis device.

The setup prepares everything needed for evaluation:

- downloads a `YOLOv8n-seg` ONNX model from HubAI
- downloads a small `COCO 2017` sample through `FiftyOne`
- parses the dataset into Luxonis Data Format (`LDF`) with `luxonis_ml`
- runs instance segmentation evaluation with `ONNX Runtime` on CPU

## Prerequisites

Install `luxonis_eval` from the repository root (`luxonis_ml` and other dependencies are pulled in automatically), and `fiftyone` for dataset handling:

```bash
# luxonis_eval install
pip install .

# install fiftyone
pip install fiftyone
```

The setup script also expects the following tools to be available in `PATH`:

- `wget`
- `tar`
- `fiftyone`

## Prepare the Model and Dataset

From the repository root, run:

```bash
bash examples/quickstart_inst_seg/setup_example.sh
```

This script:

1. fetches a fresh signed download URL for the `YOLOv8n-seg` ONNX model from HubAI
2. downloads and extracts the model into `examples/quickstart_inst_seg/models/`
3. downloads small `COCO 2017` splits with `FiftyOne`
4. parses the dataset into `LDF` as `coco-2017`

To force the model to be downloaded again, run:

```bash
bash examples/quickstart_inst_seg/setup_example.sh --force
```

## Run Evaluation

After the setup completes, run:

```bash
luxonis_eval eval --config configs/yolov8n_inst_seg_config.yaml
```

This evaluates the model on the prepared dataset using `ONNX Runtime` with the CPU execution provider.

## What the Setup Script Does

**Model download**

The script calls the HubAI API to retrieve a fresh signed URL for the `YOLOv8n-seg` ONNX archive, downloads it, extracts `yolov8n-seg.onnx`, and stores it under `examples/quickstart_inst_seg/models/`. If the model already exists locally, the download step is skipped unless `--force` is passed.

**Dataset download**

The script downloads small `COCO 2017` splits through `FiftyOne` to keep the example lightweight. By default, the downloaded data is stored under `~/fiftyone/coco-2017/`.

**LDF parsing**

Once the dataset is available locally, the script runs `luxonis_ml data parse` and creates an `LDF` dataset named `coco-2017`, which is then used by the evaluation config.
