# topsbot_ppocr

[中文文档](README_cn.md)

# Overview

`topsbot_ppocr` runs PP-OCR (PaddleOCR) two-stage text detection and recognition on the MES20 board, supporting Chinese and English scene text recognition.

- **With a camera**: real-time detection and recognition; detection results are published
- **Single image**: offline detection on a local JPG; outputs an annotated image
- **Browser preview**: use with [`topsbot_webviz`](../topsbot_webviz) to view the feed and OCR boxes in a browser

# Bill of Materials

| # | Item | Notes |
| - | ---- | ----- |
| 1 | TOPSBOT MES20 board | NPU (ta_runtime) and dmabuf environment |
| 2 | USB camera | For joint testing with `topsbot_usb_cam`; MJPEG 640×480 recommended |
| 3 | PP-OCR models | See model table below |
| 4 | `topsbot_webviz` | Web preview for joint testing (optional) |

# Build and Install

## PC (cross-compile)

Build inside the `topsbot_cc` toolchain environment to produce `install/` for the MES20:

```bash
# PC: topsbot_cc toolchain
cd topsbot_cc/topsbot_ws
./tools/build.sh -s topsbot_ppocr
```

Sync `install/` to the board, or build directly on the board if a full build environment is available.

## MES20 board (install and source)

**Option A: DEB packages (recommended when available)**

```bash
# MES20
sudo apt update
sudo apt install -y tbros-humble-topsbot-ros
```

**Option B: workspace overlay**

```bash
# MES20
source /opt/tbros/humble/setup.bash
#source <install>/setup.bash
```

## Model files

Models (`.nb`) are not shipped with source. On first launch on the board, `~/topsbot/models/` is checked; missing files trigger an automatic download of `models.zip` per `config/models_manifest.yaml`, then extraction.

| Model | Default path | Input path |
| ----- | ------------ | ---------- |
| PP-OCR Detection | `~/topsbot/models/ppocr_det_int16.nb` | INT16 |
| PP-OCR Recognition | `~/topsbot/models/ppocr_rec_float16.nb` | FP16 |

Offline or pre-installed models:

```bash
# MES20
ros2 launch topsbot_ppocr file_cli.launch.py auto_download_models:=false
```

Manual download (maintainers must set `models_zip_url` in `models_manifest.yaml`):

```bash
# PC or MES20 (source tree)
./scripts/download_models.sh
```

# Usage

Scene configs are full YAML files under `config/`.

## File mode (offline single JPG)

Use when you need a quick model check without a camera. Launch: `file_cli.launch.py`.

**Default launch**

```bash
# MES20
ros2 launch topsbot_ppocr file_cli.launch.py
```

| Item | Default |
|------|---------|
| Model | PP-OCR (`file_ppocr.yaml`) |
| Input | `share/topsbot_ppocr/imgs/test.jpg` |
| Output | `/tmp/topsbot_ppocr/out_ppocr.jpg` |

**One command for input / output** (include only what you need)

```bash
# MES20
ros2 launch topsbot_ppocr file_cli.launch.py \
  image_file:=/path/to/in.jpg \
  output_image_file:=/path/to/out.jpg
```

| Launch argument | Purpose |
|-----------------|---------|
| `image_file` | Input JPG path |
| `output_image_file` | Output JPG path |

## Single-node mode (online inference)

Use when an upstream node already publishes images (e.g. another `topsbot_usb_cam` or a custom node). Launch: `ppocr_node.launch.py`.

**Default launch** (subscribes `/tbmem_img`, publishes `/ocr`)

```bash
# MES20
ros2 launch topsbot_ppocr ppocr_node.launch.py
```

**Common adjustments**

```bash
# MES20 — multi-camera / different topic
ros2 launch topsbot_ppocr ppocr_node.launch.py input_topic:=/cam0/tbmem_img

# MES20 — frame skip
ros2 launch topsbot_ppocr ppocr_node.launch.py frame_skip:=2

# MES20 — sensor_msgs path (pair with usb_cam)
ros2 launch topsbot_usb_cam usb_cam.launch.py params_file:=mjpeg2nv12_640x480_sensor.yaml setup_shm:=false
ros2 launch topsbot_ppocr ppocr_node.launch.py input_msg:=sensor_image
```

| Launch argument | Purpose |
|-----------------|---------|
| `params_file` | Scene YAML (model, `input_msg`, topics, thresholds, …) |
| `input_topic` | Override subscription topic |
| `input_msg` | Override input type: `tb_img` / `sensor_image` / `file` |
| `frame_skip` | Frame skip; `0` means no skip |

List all launch arguments: `ros2 launch topsbot_ppocr ppocr_node.launch.py -s`

## Joint test mode (camera + PPOCR + Web preview)

End-to-end USB camera test. Launch: `cam_ppocr_webviz_demo.launch.py` (starts `topsbot_usb_cam`, `topsbot_ppocr`, and `topsbot_webviz`).

**Default launch**

```bash
# MES20
source /opt/tbros/humble/setup.bash
#source <install>/setup.bash
ros2 launch topsbot_ppocr cam_ppocr_webviz_demo.launch.py
```

Open `http://<board-ip>:9889/` in a browser to view the NV12 feed with OCR detection boxes overlaid.

Defaults: `usb_cam` `mjpeg2nv12_640x480_zc.yaml` + PPOCR `ai_infer_ppocr.yaml` + webviz `tb_img_ppocr.yaml`.

**Common adjustments**

```bash
# MES20 — switch model config (edit ai_infer_ppocr.yaml or pass a custom YAML)
ros2 launch topsbot_ppocr cam_ppocr_webviz_demo.launch.py \
  ppocr_params_file:=/path/to/my_ppocr.yaml

# MES20 — headless (no webviz)
ros2 launch topsbot_ppocr cam_ppocr_webviz_demo.launch.py enable_webviz:=false
```

**Verify the pipeline**

```bash
# MES20
ros2 topic hz /tbmem_img
ros2 topic hz /ocr
ros2 topic echo /ocr --once
```

On success, the PPOCR node logs look like:

```text
[ai_infer_node]: loaded ppocr backend on NPU core 0
[ai_infer_node]: unified AI node ready: backend=ppocr model=ppocr input=tb_img:/tbmem_img workers=1 strategy=least_loaded queue=2 drop=drop_oldest ocr=/ocr
[ai_infer_node]: AI worker core=0 healthy=true in_flight=.. assigned=.. completed=.. failed=.. dropped=.. latency_ewma=..ms
```

# Interface Reference

## ROS topics

| Direction | Topic | Message type | Notes |
|-----------|-------|--------------|-------|
| Subscribe | `/tbmem_img` (default) | `tb_img_msgs/TbMsg480P` etc. | `input_msg: tb_img` |
| Subscribe | `/image_raw` | `sensor_msgs/Image` | `input_msg: sensor_image` |
| Publish | `/ocr` | `tb_ocr_msgs/TbOcrTargets` | Boxes in original image space |

File mode writes JPG to `output_image_file` instead of publishing ROS topics.

## Node parameters

Field dictionary: [`config/params_reference.yaml`](config/params_reference.yaml) (reference only, not auto-loaded). Grouped summary:

| Group | Key parameters | Notes |
|-------|----------------|-------|
| Input (online) | `input_msg`, `input_topic`, `tb_img_profile` | `tb_img` / `sensor_image`; `tb_img_profile` applies only when `input_msg` is `tb_img` |
| Input (file) | `image_file`, `output_image_file` | Offline JPG paths |
| Output | `publish_ocr`, `ocr_topic`, `frame_skip` | OCR topic and frame skip |
| Model | `model.backend`, `model.name`, `model.type` | Backend (`ppocr`), model name and type |
| Detection | `ppocr.det_model_path`, `det_input_width`, `det_db_thresh`, … | Detection model path, input size, DB thresholds |
| Recognition | `ppocr.rec_model_path`, `rec_input_width`, `rec_thresh`, … | Recognition model path, input size, confidence threshold |
| Dictionary | `ppocr.dict_path` | Dictionary file path (`ppocr_keys_v1.txt`) |
| Debug | `enable_profile` | Print pre / infer / post timing |

Scene YAML files are full instances of the above — one file per use case:

| `params_file` | `input_msg` | Notes |
|---------------|-------------|-------|
| `ai_infer_ppocr.yaml` | `tb_img` | Joint test default (online) |
| `file_ppocr.yaml` | `file` | Offline default |

Merge order: `params_file` YAML → launch overrides (`input_topic`, `frame_skip`, etc.; non-empty only). See `config/README.md`.

## Launch files

| Launch | Purpose |
|--------|---------|
| `file_cli.launch.py` | Offline file mode |
| `ppocr_node.launch.py` | Online single node |
| `cam_ppocr_webviz_demo.launch.py` | Camera + PPOCR + Web joint test |

List arguments: `ros2 launch topsbot_ppocr <launch> -s`


# License

Apache License 2.0 — see [LICENSE](./LICENSE) and [NOTICE](./NOTICE).

Contributions welcome; see [CONTRIBUTING.md](./CONTRIBUTING.md).
