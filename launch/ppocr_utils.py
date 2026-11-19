# Copyright 2026 TOPSBOT
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import yaml
from ament_index_python.packages import get_package_share_directory


def resolve_params_file(package_share: str, params_file: str) -> str:
    path = params_file if os.path.isabs(params_file) else os.path.join(
        package_share, "config", params_file)
    if not os.path.isfile(path):
        raise RuntimeError(f"topsbot_ppocr params file not found: {path}")
    return path


def load_params(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data.get("/**", {}).get("ros__parameters", data)


def param_value(params: dict, key: str, default=None):
    if key in params:
        return params[key]
    value = params
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def resolve_resource_path(path: str) -> str:
    if not path.startswith("package://"):
        return os.path.expanduser(path)
    resource = path[len("package://"):]
    package, separator, relative = resource.partition("/")
    if not package or not separator or not relative:
        raise RuntimeError(f"invalid package resource path: {path}")
    return os.path.join(get_package_share_directory(package), relative)


def ensure_ppocr_models(context, params: dict, models_dir: str = "~/topsbot/models") -> tuple:
    """Check and return detector, recognizer model paths, and dictionary path."""
    package_share = get_package_share_directory("topsbot_ppocr")
    script = os.path.join(package_share, "scripts", "ensure_models.sh")

    models_dir_expanded = os.path.expanduser(models_dir)
    dataset_dir = os.path.expanduser("~/topsbot/dataset/ppocr")

    det_model = os.path.expanduser(str(param_value(
        params, "ppocr.det_model_path", "~/topsbot/models/ppocr_det_int16.nb")))
    rec_model = os.path.expanduser(str(param_value(
        params, "ppocr.rec_model_path", "~/topsbot/models/ppocr_rec_float16.nb")))
    dict_path = os.path.expanduser(str(param_value(
        params, "ppocr.dict_path", "~/topsbot/dataset/ppocr/ppocr_keys_v1.txt")))

    # Check if all files exist
    all_exist = os.path.isfile(det_model) and os.path.isfile(rec_model) and os.path.isfile(dict_path)

    if not all_exist:
        # Run ensure_models script (will auto-download)
        if os.path.isfile(script):
            print(f"[ppocr_utils] Running ensure_models script: {script}")
            result = subprocess.run(
                [script, "--models-dir", models_dir_expanded, "--auto-download", "true"],
                check=False,
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print(result.stdout)
                print(result.stderr)
                raise RuntimeError(
                    f"PP-OCR resources not found or download failed.\n"
                    f"Please run manually: bash {script}")
            print(result.stdout)
        else:
            raise RuntimeError(f"ensure_models.sh script not found: {script}")

    # Re-check after script
    if not os.path.isfile(det_model):
        raise RuntimeError(f"PP-OCR detector model not found: {det_model}")
    if not os.path.isfile(rec_model):
        raise RuntimeError(f"PP-OCR recognizer model not found: {rec_model}")
    if not os.path.isfile(dict_path):
        raise RuntimeError(f"PP-OCR dictionary not found: {dict_path}")

    return det_model, rec_model, dict_path

