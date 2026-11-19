#!/bin/bash
# Copyright 2026 TOPSBOT
# SPDX-License-Identifier: Apache-2.0
#
# PP-OCR resources download script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="${HOME}/topsbot/dataset/ppocr"
MODELS_DIR="${HOME}/topsbot/models"

# Aliyun OSS base URL (adjust to your actual OSS bucket)
OSS_BASE_URL="https://topsbot-models.oss-cn-beijing.aliyuncs.com/ppocr"

usage() {
    cat << EOF
Usage: $0 [options]

Options:
    --dataset-dir DIR   Dataset directory (default: ~/topsbot/dataset/ppocr)
    --models-dir DIR    Models directory (default: ~/topsbot/models)
    --force            Force re-download even if files exist
    --help             Show this help message

This script downloads PP-OCR resources:
- Dictionary file (ppocr_keys_v1.txt)
- Pre-compiled .nb models for TOPSBOT NPU
- Sample test images
EOF
}

FORCE_DOWNLOAD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset-dir)
            DATASET_DIR="$2"
            shift 2
            ;;
        --models-dir)
            MODELS_DIR="$2"
            shift 2
            ;;
        --force)
            FORCE_DOWNLOAD=true
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

mkdir -p "${DATASET_DIR}"
mkdir -p "${MODELS_DIR}"

echo "PP-OCR Resources Download"
echo "========================="
echo "Dataset directory: ${DATASET_DIR}"
echo "Models directory:  ${MODELS_DIR}"
echo ""

# Download dictionary file
DICT_FILE="${DATASET_DIR}/ppocr_keys_v1.txt"
if [[ ! -f "${DICT_FILE}" ]] || [[ "${FORCE_DOWNLOAD}" == "true" ]]; then
    echo "Downloading dictionary file..."
    if command -v wget &> /dev/null; then
        wget -O "${DICT_FILE}" "${OSS_BASE_URL}/ppocr_keys_v1.txt" || {
            echo "Warning: Failed to download from OSS, using fallback URL"
            wget -O "${DICT_FILE}" "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/ppocr/utils/ppocr_keys_v1.txt"
        }
    elif command -v curl &> /dev/null; then
        curl -L -o "${DICT_FILE}" "${OSS_BASE_URL}/ppocr_keys_v1.txt" || {
            echo "Warning: Failed to download from OSS, using fallback URL"
            curl -L -o "${DICT_FILE}" "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/ppocr/utils/ppocr_keys_v1.txt"
        }
    else
        echo "Error: Neither wget nor curl found. Cannot download files."
        exit 1
    fi
    echo "✓ Dictionary downloaded to ${DICT_FILE}"
else
    echo "✓ Dictionary already exists: ${DICT_FILE}"
fi

# Download detector model
DET_MODEL="${MODELS_DIR}/ppocr_det_int16.nb"
if [[ ! -f "${DET_MODEL}" ]] || [[ "${FORCE_DOWNLOAD}" == "true" ]]; then
    echo "Downloading detector model (this may take a while)..."
    if command -v wget &> /dev/null; then
        wget -O "${DET_MODEL}" "${OSS_BASE_URL}/ppocr_det_int16.nb" || {
            echo "Error: Failed to download detector model from ${OSS_BASE_URL}"
            echo "Please manually place the model at: ${DET_MODEL}"
            exit 1
        }
    elif command -v curl &> /dev/null; then
        curl -L -o "${DET_MODEL}" "${OSS_BASE_URL}/ppocr_det_int16.nb" || {
            echo "Error: Failed to download detector model from ${OSS_BASE_URL}"
            echo "Please manually place the model at: ${DET_MODEL}"
            exit 1
        }
    fi
    echo "✓ Detector model downloaded to ${DET_MODEL}"
else
    echo "✓ Detector model already exists: ${DET_MODEL}"
fi

# Download recognizer model
REC_MODEL="${MODELS_DIR}/ppocr_rec_float16.nb"
if [[ ! -f "${REC_MODEL}" ]] || [[ "${FORCE_DOWNLOAD}" == "true" ]]; then
    echo "Downloading recognizer model (this may take a while)..."
    if command -v wget &> /dev/null; then
        wget -O "${REC_MODEL}" "${OSS_BASE_URL}/ppocr_rec_float16.nb" || {
            echo "Error: Failed to download recognizer model from ${OSS_BASE_URL}"
            echo "Please manually place the model at: ${REC_MODEL}"
            exit 1
        }
    elif command -v curl &> /dev/null; then
        curl -L -o "${REC_MODEL}" "${OSS_BASE_URL}/ppocr_rec_float16.nb" || {
            echo "Error: Failed to download recognizer model from ${OSS_BASE_URL}"
            echo "Please manually place the model at: ${REC_MODEL}"
            exit 1
        }
    fi
    echo "✓ Recognizer model downloaded to ${REC_MODEL}"
else
    echo "✓ Recognizer model already exists: ${REC_MODEL}"
fi

# Download sample test images
SAMPLE_DIR="${DATASET_DIR}/samples"
mkdir -p "${SAMPLE_DIR}"

SAMPLE_IMAGES=("chinese_text.jpg" "receipt.jpg" "document.jpg")
for img in "${SAMPLE_IMAGES[@]}"; do
    SAMPLE_FILE="${SAMPLE_DIR}/${img}"
    if [[ ! -f "${SAMPLE_FILE}" ]] || [[ "${FORCE_DOWNLOAD}" == "true" ]]; then
        echo "Downloading sample image: ${img}..."
        wget -O "${SAMPLE_FILE}" "${OSS_BASE_URL}/samples/${img}" 2>/dev/null || \
        curl -L -o "${SAMPLE_FILE}" "${OSS_BASE_URL}/samples/${img}" 2>/dev/null || \
        echo "  (Sample image ${img} not available, skipping)"
    fi
done

echo ""
echo "✓ PP-OCR resources download complete!"
echo ""
echo "Files downloaded:"
echo "  Dictionary:  ${DICT_FILE}"
echo "  Detector:    ${DET_MODEL}"
echo "  Recognizer:  ${REC_MODEL}"
echo "  Samples:     ${SAMPLE_DIR}/"
echo ""
echo "You can now run PP-OCR inference with:"
echo "  ros2 launch topsbot_ppocr file_cli.launch.py image_file:=/path/to/image.jpg"

exit 0
