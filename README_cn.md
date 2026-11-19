# topsbot_ppocr

[English](README.md)

# 概述

`topsbot_ppocr` 在 MES20 开发板上运行 PP-OCR（PaddleOCR）文字检测与识别，支持中英文场景文字识别。

- **相机模式**：实时检测与识别；结果检测发布
- **单图模式**：离线检测本地 JPG 图片；输出标注图像
- **浏览器预览**：配合 [`topsbot_webviz`](../topsbot_webviz) 在浏览器中查看视频流与检测框

# 物料清单

| # | 项目 | 说明 |
| - | ---- | ----- |
| 1 | TOPSBOT MES20 开发板 | NPU（ta_runtime）与 dmabuf 环境 |
| 2 | USB 摄像头 | 配合 `topsbot_usb_cam` 联合测试；推荐 MJPEG 640×480 |
| 3 | PP-OCR 模型 | 见下方模型表 |
| 4 | `topsbot_webviz` | Web 预览联合测试（可选） |

# 编译与安装

## PC（交叉编译）

在 `topsbot_cc` 工具链环境中编译，生成供 MES20 使用的 `install/`：

```bash
# PC: topsbot_cc 工具链
cd topsbot_cc/topsbot_ws
./tools/build.sh -s topsbot_ppocr
```

将 `install/` 同步到开发板，或在开发板上直接编译（如果有完整编译环境）。

## MES20 开发板（安装与环境配置）

**方案 A：DEB 包（推荐）**

```bash
# MES20
sudo apt update
sudo apt install -y tbros-humble-topsbot-ros
```

**方案 B：工作空间覆盖**

```bash
# MES20
source /opt/tbros/humble/setup.bash
#source <install>/setup.bash
```

## 模型文件

模型（`.nb`）不随源码发布。在开发板上首次启动时，会检查 `~/topsbot/models/`；缺失文件会自动下载 `models.zip`（按 `config/models_manifest.yaml` 配置），然后解压。

| 模型 | 默认路径 | 输入路径 |
| ---- | -------- | -------- | -------- |
| PP-OCR 检测 | `~/topsbot/models/ppocr_det_int16.nb` | INT16 |
| PP-OCR 识别 | `~/topsbot/models/ppocr_rec_float16.nb` | FP16 |

离线或预装模型：

```bash
# MES20
ros2 launch topsbot_ppocr file_cli.launch.py auto_download_models:=false
```

手动下载（维护者需在 `models_manifest.yaml` 中设置 `models_zip_url`）：

```bash
# PC 或 MES20（源码树）
./scripts/download_models.sh
```

# 使用方法

场景配置文件是 `config/` 下的完整 YAML 文件。

## 文件模式（离线单张 JPG）

无需相机即可快速验证模型。启动文件：`file_cli.launch.py`。

**默认启动**

```bash
# MES20
ros2 launch topsbot_ppocr file_cli.launch.py
```

| 项目 | 默认值 |
|------|--------|
| 模型 | PP-OCR（`file_ppocr.yaml`） |
| 输入 | `share/topsbot_ppocr/imgs/test.jpg` |
| 输出 | `/tmp/topsbot_ppocr/out_ppocr.jpg` |

**一条命令指定模型 / 输入 / 输出**（按需包含）

```bash
# MES20
ros2 launch topsbot_ppocr file_cli.launch.py \
  image_file:=/path/to/in.jpg \
  output_image_file:=/path/to/out.jpg
```

| 启动参数 | 用途 |
|---------|------|
| `image_file` | 输入 JPG 路径 |
| `output_image_file` | 输出 JPG 路径 |

## 单节点模式（在线推理）

当上游节点已发布图像时使用（如另一个 `topsbot_usb_cam` 或自定义节点）。启动文件：`ppocr_node.launch.py`。

**默认启动**（订阅 `/tbmem_img`，发布 `/ocr`）

```bash
# MES20
ros2 launch topsbot_ppocr ppocr_node.launch.py
```

**常见调整**

```bash
# MES20 — 多相机 / 不同话题
ros2 launch topsbot_ppocr ppocr_node.launch.py input_topic:=/cam0/tbmem_img

# MES20 — 跳帧
ros2 launch topsbot_ppocr ppocr_node.launch.py frame_skip:=2

# MES20 — sensor_msgs 路径（配合 usb_cam）
ros2 launch topsbot_usb_cam usb_cam.launch.py params_file:=mjpeg2nv12_640x480_sensor.yaml setup_shm:=false
ros2 launch topsbot_ppocr ppocr_node.launch.py input_msg:=sensor_image
```

| 启动参数 | 用途 |
|---------|------|
| `params_file` | 场景 YAML（模型、`input_msg`、话题、阈值等） |
| `input_topic` | 覆盖订阅话题 |
| `input_msg` | 覆盖输入类型：`tb_img` / `sensor_image` / `file` |
| `frame_skip` | 跳帧；`0` 表示不跳帧 |

列出所有启动参数：`ros2 launch topsbot_ppocr ppocr_node.launch.py -s`

## 联合测试模式（相机 + PPOCR + Web 预览）

端到端 USB 摄像头测试。启动文件：`cam_ppocr_webviz_demo.launch.py`（启动 `topsbot_usb_cam`、`topsbot_ppocr` 和 `topsbot_webviz`）。

**默认启动**

```bash
# MES20
source /opt/tbros/humble/setup.bash
#source <install>/setup.bash
ros2 launch topsbot_ppocr cam_ppocr_webviz_demo.launch.py
```

在浏览器中打开 `http://<board-ip>:9889/` 查看带检测框覆盖的 NV12 视频流。

默认配置：`usb_cam` `mjpeg2nv12_640x480_zc.yaml` + PPOCR `ai_infer_ppocr.yaml` + webviz `tb_img_ppocr.yaml`。

**常见调整**

```bash
# MES20 — 切换模型配置（编辑 ai_infer_ppocr.yaml 或传入自定义 YAML）
ros2 launch topsbot_ppocr cam_ppocr_webviz_demo.launch.py \
  ppocr_params_file:=/path/to/my_ppocr.yaml

# MES20 — 无头模式（无 webviz）
ros2 launch topsbot_ppocr cam_ppocr_webviz_demo.launch.py enable_webviz:=false
```

**验证流水线**

```bash
# MES20
ros2 topic hz /tbmem_img
ros2 topic hz /ocr
ros2 topic echo /ocr --once
```

成功时，PPOCR 节点日志如下：

```text
[ai_infer_node]: loaded ppocr backend on NPU core 0
[ai_infer_node]: unified AI node ready: backend=ppocr model=ppocr input=tb_img:/tbmem_img workers=1 strategy=least_loaded queue=2 drop=drop_oldest ocr=/ocr
[ai_infer_node]: AI worker core=0 healthy=true in_flight=.. assigned=.. completed=.. failed=.. dropped=.. latency_ewma=..ms
```

# 接口参考

## ROS 话题

| 方向 | 话题 | 消息类型 | 说明 |
|------|------|----------|------|
| 订阅 | `/tbmem_img`（默认） | `tb_img_msgs/TbMsg480P` 等 | `input_msg: tb_img` |
| 订阅 | `/image_raw` | `sensor_msgs/Image` | `input_msg: sensor_image` |
| 发布 | `/ocr` | `tb_ocr_msgs/TbOcrTargets` | 检测框位于原图坐标系 |

文件模式将 JPG 写入 `output_image_file`，而非发布 ROS 话题。

## 节点参数

字段字典：[`config/params_reference.yaml`](config/params_reference.yaml)（仅供参考，不自动加载）。分组汇总：

| 分组 | 关键参数 | 说明 |
|------|---------|------|
| 输入（在线） | `input_msg`、`input_topic`、`tb_img_profile` | `tb_img` / `sensor_image`；`tb_img_profile` 仅在 `input_msg` 为 `tb_img` 时生效 |
| 输入（文件） | `image_file`、`output_image_file` | 离线 JPG 路径 |
| 输出 | `publish_ocr`、`ocr_topic`、`frame_skip` | OCR 话题与跳帧 |
| 模型 | `model.backend`、`model.name`、`model.type` | 后端（`ppocr`）、模型名称与类型 |
| 检测 | `ppocr.det_model_path`、`det_input_width`、`det_db_thresh` 等 | 检测模型路径、输入尺寸、DB 阈值 |
| 识别 | `ppocr.rec_model_path`、`rec_input_width`、`rec_thresh` 等 | 识别模型路径、输入尺寸、置信度阈值 |
| 字典 | `ppocr.dict_path` | 字典文件路径（`ppocr_keys_v1.txt`） |
| 调试 | `enable_profile` | 打印前处理 / 推理 / 后处理耗时 |

场景 YAML 文件是上述参数的完整实例——每个用例一个文件：

| `params_file` | `input_msg` | 说明 |
|---------------|-------------|------|
| `ai_infer_ppocr.yaml` | `tb_img` | 联合测试默认（在线） |
| `file_ppocr.yaml` | `file` | 离线默认 |

合并顺序：`params_file` YAML → 启动覆盖（`input_topic`、`frame_skip` 等；仅非空）。详见 `config/README.md`。

## 启动文件

| 启动文件 | 用途 |
|---------|------|
| `file_cli.launch.py` | 离线文件模式 |
| `ppocr_node.launch.py` | 在线单节点 |
| `cam_ppocr_webviz_demo.launch.py` | 相机 + PPOCR + Web 联合测试 |

列出参数：`ros2 launch topsbot_ppocr <launch> -s`


# 许可证

Apache License 2.0 — 见 [LICENSE](./LICENSE) 和 [NOTICE](./NOTICE)。

欢迎贡献；见 [CONTRIBUTING.md](./CONTRIBUTING.md)。

