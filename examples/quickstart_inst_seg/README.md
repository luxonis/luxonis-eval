# Quickstart E2E Example: YOLOv8 Instance Segmentation (ONNX)

This example provides a minimal end-to-end run of `luxonis-eval` without a Luxonis device:

- Downloads a YOLOv8 instance segmentation ONNX model from HubAI
- Downloads small COCO-2017 splits with FiftyOne
- Converts the dataset into LDF with `luxonis_ml data parse`
- Runs evaluation through ONNX Runtime on CPU

## Prerequisites

Install `luxonis_eval` from the repository root (`luxonis_ml` and other dependencies are pulled in automatically):

```bash
# standard install
pip install .
```

The setup script also requires `wget` and `tar` to be available on your system.

## 1. Prepare Model + Dataset

Activate your Python environment, then from the repository root run:

```bash
bash examples/quickstart_inst_seg/setup_example.sh
```

This will:

1. Fetch a fresh download URL for the YOLOv8n-seg ONNX model from HubAI and download it
2. Download small COCO-2017 splits (test/train: 10 samples, validation: 1000 samples) via FiftyOne
3. Parse the dataset into LDF format with `luxonis_ml`

To force a re-download of the model even if it already exists locally:

```bash
bash examples/quickstart_inst_seg/setup_example.sh --force
```

## 2. Run Evaluation

```bash
luxonis_eval eval --config configs/yolov8n_inst_seg_config.yaml
```

This runs instance segmentation metrics with ONNX Runtime on CPU.

## Setup script walkthrough

**Model:** Calls the HubAI API to get a fresh signed download URL for the YOLOv8n-seg ONNX model, downloads and extracts it into `examples/quickstart_inst_seg/models/`. If the model file already exists it skips the download unless `--force` is passed.

**Dataset:** Downloads three COCO-2017 splits via FiftyOne (test, train, validation) with a small sample cap to keep things fast. The downloaded data lands in `~/fiftyone/coco-2017/` by default.

**LDF parse:** Converts the downloaded COCO data into LDF format using `luxonis_ml data parse`, creating an LDF dataset named `coco-2017`, which is what `luxonis_eval` expects as input for evaluation.
