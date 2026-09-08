#!/usr/bin/env bash
set -euo pipefail

DATASET_NAME="coco-2017"
LDF_DATASET_NAME="quickstartcoco"
MODEL_API_BASE="https://easyml.cloud.luxonis.com/models/api/v1/modelInstances"
ONNX_MODEL_INSTANCE_ID="aimi_B5kAk351EfuabVbYkJBror"
RVC4_MODEL_INSTANCE_ID="aimi_CbmxoCSsP6dHmX9XKcjLZL"

die() {
  echo "Error: $*" >&2
  exit 1
}

require() {
  command -v "$1" >/dev/null 2>&1 || die "'$1' not found in PATH."
}

download_model() {
  local instance_id="$1"
  local model_path="$2"
  local model_name="$3"

  if [[ -f "${model_path}" ]]; then
    echo "${model_name} model already exists: ${model_path}"
    return
  fi

  echo "Fetching ${model_name} download URL from HubAI..."
  local api_response
  api_response="$(wget -O- "${MODEL_API_BASE}/${instance_id}/download")"
  [[ -n "${api_response}" ]] \
    || die "Failed to fetch the ${model_name} download URL from HubAI."

  local download_url
  download_url="$(echo "${api_response}" | tr -d '[]"' | tr ',' '\n' | head -n 1)"
  [[ -n "${download_url}" ]] \
    || die "Could not parse the ${model_name} download URL."

  local temporary_path="${model_path}.download"
  echo "Downloading ${model_name} model..."
  if ! wget -O "${temporary_path}" "${download_url}"; then
    rm -f "${temporary_path}"
    die "Failed to download the ${model_name} model."
  fi
  mv "${temporary_path}" "${model_path}"
  echo "${model_name} model saved: ${model_path}"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
QUICKSTART_DATA_DIR="${QUICKSTART_DATA_DIR:-${REPO_ROOT}/.cache/quickstart/fiftyone}"
export FIFTYONE_DATASET_ZOO_DIR="${QUICKSTART_DATA_DIR}"

require wget
require fiftyone
require luxonis_ml

download_model \
  "${ONNX_MODEL_INSTANCE_ID}" \
  "${SCRIPT_DIR}/yolov6.onnx.tar" \
  "ONNX"
download_model \
  "${RVC4_MODEL_INSTANCE_ID}" \
  "${SCRIPT_DIR}/yolov6.rvc4.tar" \
  "RVC4"

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
