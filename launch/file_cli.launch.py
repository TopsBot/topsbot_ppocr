# Copyright 2026 TOPSBOT
# SPDX-License-Identifier: Apache-2.0

import os
import sys

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

_launch_dir = os.path.dirname(os.path.abspath(__file__))
if _launch_dir not in sys.path:
    sys.path.insert(0, _launch_dir)
from ppocr_utils import (  # noqa: E402
    load_params,
    param_value,
    resolve_params_file,
    resolve_resource_path,
    ensure_ppocr_models,
)


def _launch_setup(context):
    package_share = get_package_share_directory("topsbot_ppocr")
    params_path = resolve_params_file(
        package_share, LaunchConfiguration("params_file").perform(context))
    params = load_params(params_path)

    models_dir = LaunchConfiguration("models_dir").perform(context)
    det_model, rec_model, dict_path = ensure_ppocr_models(context, params, models_dir)

    image_file = LaunchConfiguration("image_file").perform(context)
    output_file = LaunchConfiguration("output_image_file").perform(context) or str(
        param_value(params, "output_image_file", "/tmp/topsbot_ai_infer/out_ppocr.jpg"))

    executable = os.path.join(
        get_package_prefix("topsbot_ppocr"),
        "lib", "topsbot_ppocr", "ppocr_file_cli")
    command = [
        executable,
        image_file,
        output_file,
        "--det-model", det_model,
        "--rec-model", rec_model,
        "--dict-path", dict_path,
        "--device-id", str(int(param_value(params, "device_id", 0))),
        "--det-db-thresh", str(float(param_value(params, "ppocr.det_db_thresh", 0.3))),
        "--det-box-thresh", str(float(param_value(params, "ppocr.det_db_box_thresh", 0.6))),
        "--rec-thresh", str(float(param_value(params, "ppocr.rec_thresh", 0.5))),
    ]
    return [
        LogInfo(msg=f"[ppocr_file_cli] params_file: {params_path}"),
        ExecuteProcess(cmd=command, output="screen"),
    ]


def generate_launch_description():
    default_image = PathJoinSubstitution([
        FindPackageShare("topsbot_ppocr"),
        "imgs",
        "test.jpg",
    ])

    return LaunchDescription([
        DeclareLaunchArgument("params_file", default_value="file_ppocr.yaml"),
        DeclareLaunchArgument(
            "image_file",
            default_value=default_image,
            description="Input JPEG (default: share/topsbot_ppocr/imgs/test.jpg)",
        ),
        DeclareLaunchArgument("output_image_file", default_value=""),
        DeclareLaunchArgument("models_dir", default_value="~/topsbot/models"),
        OpaqueFunction(function=_launch_setup),
    ])
