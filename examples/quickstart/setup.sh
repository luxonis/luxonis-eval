#!/usr/bin/env bash
set -euo pipefail

DATASET_NAME="coco-2017"
LDF_DATASET_NAME="coco-2017-quickstart"

die() {
  echo "Error: $*" >&2
  exit 1
}

require() {
  command -v "$1" >/dev/null 2>&1 || die "'$1' not found in PATH."
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
QUICKSTART_DATA_DIR="${QUICKSTART_DATA_DIR:-${REPO_ROOT}/.cache/quickstart/fiftyone}"
export FIFTYONE_DATASET_ZOO_DIR="${QUICKSTART_DATA_DIR}"

require fiftyone
require luxonis_ml

[[ -f "${REPO_ROOT}/yolov6.onnx.tar" ]] \
  || die "Missing model archive: ${REPO_ROOT}/yolov6.onnx.tar"

mkdir -p "${FIFTYONE_DATASET_ZOO_DIR}"

echo "Downloading 800 COCO training images..."
fiftyone zoo datasets load "${DATASET_NAME}" \
  --split train \
  --kwargs max_samples=800 label_types=detections

echo "Downloading 100 COCO validation images..."
fiftyone zoo datasets load "${DATASET_NAME}" \
  --split validation \
  --kwargs max_samples=100 label_types=detections

echo "Downloading 100 COCO test images..."
fiftyone zoo datasets load "${DATASET_NAME}" \
  --split test \
  --kwargs max_samples=100 label_types=detections

COCO_ROOT="${FIFTYONE_DATASET_ZOO_DIR}/${DATASET_NAME}"
[[ -d "${COCO_ROOT}" ]] || die "COCO dataset not found at: ${COCO_ROOT}"

echo "Parsing COCO into Luxonis Data Format..."
luxonis_ml data parse --delete "${COCO_ROOT}" \
  --name "${LDF_DATASET_NAME}" \
  --train 800 \
  --val 100 \
  --test 100

echo
echo "Setup complete. Run ONNX evaluation from the repository root:"
echo "  luxonis_eval eval --config examples/quickstart/onnx_config.yaml"
