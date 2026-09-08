# Quickstart: evaluate YOLOv6 on COCO

This example downloads a 1,000-image subset of COCO 2017, converts it to
Luxonis Data Format (LDF), and evaluates a YOLOv6 model. ONNX evaluation runs
locally. Optionally, evaluation can be run on a Luxonis device if you have one.

The downloaded images are split as follows:

| COCO split | LDF split | Images | Used for evaluation |
| --- | --- | ---: | --- |
| `train` | `train` | 800 | No |
| `validation` | `val` | 100 | Yes |
| `test` | `test` | 100 | No |

COCO 2017 does not publish annotations for its test split, so this example
evaluates the model on `val`.

## 1. Install the packages

From the repository root, create and activate a Python 3.10+ virtual
environment, then install `luxonis-eval` and FiftyOne:

```bash
pip install .
pip install fiftyone
```

Installing this repository also installs `luxonis-ml`, whose CLI is used to
parse COCO into LDF (Luxonis-internal dataset representation format).

## 2. Download the models and prepare COCO

Run the setup script from the repository root:

```bash
bash examples/quickstart/setup.sh
```

The script downloads the `R2 COCO 512x384` ONNX and RVC4 NNArchives from the
[YOLOv6 Nano page on models.luxonis.com](https://models.luxonis.com/luxonis/yolov6-nano/aim_RFFiRGVUcFAVE894kxRbRm?backTo=%2F)
through the public HubAI API. It saves them as
`examples/quickstart/yolov6.onnx.tar` and
`examples/quickstart/yolov6.rvc4.tar`.

It then uses the
[FiftyOne COCO-2017 CLI workflow](https://docs.voxel51.com/dataset_zoo/datasets/coco_2017.html)
to download 800 training, 100 validation, and 100 test images. It stores the
download under `.cache/quickstart/fiftyone/`, then runs:

```bash
luxonis_ml data parse --delete .cache/quickstart/fiftyone/coco-2017 \
    --name quickstartcoco \
    --train 800 --val 100 --test 100
```

The integer split sizes tell `luxonis-ml` to preserve the original COCO split
boundaries. The `--delete` flag removes an existing local LDF dataset named
`quickstartcoco` before recreating it.

To store the FiftyOne download elsewhere, set `QUICKSTART_DATA_DIR` to a clean
directory:

```bash
QUICKSTART_DATA_DIR=/path/to/fiftyone bash examples/quickstart/setup.sh
```

## 3. Evaluate with ONNX Runtime

`yolov6.onnx.tar` is an NNArchive containing the ONNX model and its
preprocessing and YOLO parser metadata. Run:

```bash
luxonis_eval eval --config examples/quickstart/onnx_config.yaml
```

No Luxonis device is required. The command uses ONNX Runtime's CPU execution
provider and reports bounding-box mean average precision on the 100 validation
images. Rendered predictions are saved under
`visualizations/quickstart/onnx/`; no window is displayed.

## 4. Optional: evaluate on a device

If an RVC4 device is connected, use the dedicated device configuration. It
selects the DepthAI engine and `yolov6.rvc4.tar`:

```bash
luxonis_eval eval --config examples/quickstart/rvc4_config.yaml
```

DepthAI discovers a connected device automatically. To select a network
device, uncomment `pipeline.engine.params` in `rvc4_config.yaml` and set its
IP address before running the same command:

```bash
luxonis_eval eval --config examples/quickstart/rvc4_config.yaml
```

Both archives describe the same YOLOv6 model, so the same dataset, parser, and
metric configuration is used for host and device evaluation. Rendered RVC4
predictions are saved under `visualizations/quickstart/rvc4/` without opening
a display window.
