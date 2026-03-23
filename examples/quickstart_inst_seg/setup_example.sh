#!/usr/bin/env bash
set -euo pipefail

MODEL_INSTANCE_ID="aimi_QtcY6CKM2cxB2QpmHVU4QL"
MODEL_API_BASE="https://easyml.cloud.luxonis.com/models/api/v1/modelInstances"
MODEL_PATH="examples/quickstart_inst_seg/models/yolov8n-seg.onnx"
ARCHIVE_PATH="examples/quickstart_inst_seg/models/yolov8n-seg.onnx.tar.xz"
DATASET_NAME="coco-2017"
FIFTYONE_ROOT="${HOME}/fiftyone"
FORCE="false"

# ── Argument parsing ──────────────────────────────────────────────────────────

for arg in "$@"; do
  case "$arg" in
    --force) FORCE="true" ;;
    *) echo "Unknown argument: $arg" >&2; exit 1 ;;
  esac
done

# ── Helpers ───────────────────────────────────────────────────────────────────

die()     { echo "Error: $*" >&2; exit 1; }
require() { command -v "$1" >/dev/null 2>&1 || die "'$1' not found in PATH."; }

# ── Model ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODEL_ABS="${REPO_ROOT}/${MODEL_PATH}"
ARCHIVE_ABS="${REPO_ROOT}/${ARCHIVE_PATH}"

if [[ -f "${MODEL_ABS}" && "${FORCE}" != "true" ]]; then
  echo "Model already exists: ${MODEL_ABS}"
else
  require wget
  require tar

  mkdir -p "$(dirname "${MODEL_ABS}")"

  echo "Fetching download URL..."
  API_RESPONSE="$(wget -O- "${MODEL_API_BASE}/${MODEL_INSTANCE_ID}/download")"
  [[ -n "${API_RESPONSE}" ]] || die "Failed to reach HubAI API. Check your connection or model instance ID."

  DOWNLOAD_URL="$(echo "${API_RESPONSE}" | tr -d '[]"' | tr ',' '\n' | head -n 1)"
  [[ -n "${DOWNLOAD_URL}" ]] || die "Could not parse download URL from API response: ${API_RESPONSE}"

  echo "Downloading model..."
  wget -O "${ARCHIVE_ABS}" "${DOWNLOAD_URL}" \
    || die "Download failed."

  echo "Extracting archive..."
  TMP_DIR="$(mktemp -d)"
  trap 'rm -rf "${TMP_DIR}" "${ARCHIVE_ABS}"' EXIT
  tar -xJf "${ARCHIVE_ABS}" -C "${TMP_DIR}"

  ONNX_FILE="$(find "${TMP_DIR}" -name 'yolov8n-seg.onnx' | head -n 1)"
  [[ -n "${ONNX_FILE}" ]] || die "Could not find 'yolov8n-seg.onnx' in archive."

  mv "${ONNX_FILE}" "${MODEL_ABS}"
  trap - EXIT
  rm -rf "${TMP_DIR}" "${ARCHIVE_ABS}"

  echo "Model saved: ${MODEL_ABS}"
fi

# ── Dataset ───────────────────────────────────────────────────────────────────

require fiftyone

for split in test train; do
  fiftyone zoo datasets load "${DATASET_NAME}" --split "${split}" \
    --kwargs max_samples=10 label_types=detections,segmentations,keypoints
done

fiftyone zoo datasets load "${DATASET_NAME}" --split validation \
  --kwargs max_samples=1000 label_types=detections,segmentations,keypoints

# ── LDF parse ─────────────────────────────────────────────────────────────────

require luxonis_ml

COCO_ROOT="${FIFTYONE_ROOT}/${DATASET_NAME}"
[[ -d "${COCO_ROOT}" ]] || die "COCO dataset not found at: ${COCO_ROOT}"

luxonis_ml data parse --delete "${COCO_ROOT}" --name "${DATASET_NAME}" --split-ratio '[0,1000,0]'

# ── Done ──────────────────────────────────────────────────────────────────────

echo ""
echo "Setup complete. Run evaluation with:"
echo "  luxonis_eval eval --config examples/quickstart_inst_seg/yolov8n_inst_seg_config.yaml"