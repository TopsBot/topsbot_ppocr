# Changelog

All notable changes to this package are documented here.

## 0.1.0 (2026-10-16)

### Added

- PP-OCR two-stage text detection (DB) and recognition (CRNN/CTC) backend plugin.
- `ppocr_file_cli` for offline JPG validation with annotated output.
- `ppocr_node.launch.py` for online single-node inference.
- `cam_ppocr_webviz_demo.launch.py` (usb_cam + ppocr + webviz) joint test pipeline.
- Full-scene yaml configs (`ai_infer_ppocr.yaml`, `file_ppocr.yaml`).
- OCR result terminal logging in `ai_infer_node.cpp`.
- Webviz overlay support with Canvas `ctx.fillText` for Chinese/English text.
- Model auto-download via `scripts/download_models.sh` and `ensure_models.sh`.
- Bilingual README (English / 中文).
