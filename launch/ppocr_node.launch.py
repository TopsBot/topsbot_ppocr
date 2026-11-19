# Copyright 2026 TOPSBOT
# SPDX-License-Identifier: Apache-2.0

import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_launch_dir = os.path.dirname(os.path.abspath(__file__))
if _launch_dir not in sys.path:
    sys.path.insert(0, _launch_dir)
from ppocr_utils import (  # noqa: E402
    ensure_ppocr_models,
    load_params,
    resolve_params_file,
)


def _launch_setup(context):
    package_share = get_package_share_directory("topsbot_ppocr")
    params_file_arg = LaunchConfiguration("params_file").perform(context)
    params_path = resolve_params_file(package_share, params_file_arg)

    models_dir = LaunchConfiguration("models_dir").perform(context)
    params = load_params(params_path)

    # Ensure models are present (auto-download if missing)
    ensure_ppocr_models(context, params, models_dir)

    return [
        LogInfo(msg=f"[ppocr_node] Starting ai_infer_node with {params_file_arg}"),
        Node(
            package="topsbot_ai_infer",
            executable="ai_infer_node",
            name="ai_infer_node",
            output="screen",
            parameters=[params_path],
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "params_file",
            default_value="ai_infer_ppocr.yaml",
            description="PP-OCR params file (default: ai_infer_ppocr.yaml)",
        ),
        DeclareLaunchArgument("models_dir", default_value="~/topsbot/models"),
        OpaqueFunction(function=_launch_setup),
    ])
