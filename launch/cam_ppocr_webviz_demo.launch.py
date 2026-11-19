# Copyright 2026 TOPSBOT contributors.
# SPDX-License-Identifier: Apache-2.0

"""Full live pipeline: usb_cam -> PPOCR -> WebViz OCR overlay."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.utilities import perform_substitutions
from launch_ros.substitutions import FindPackageShare


def _as_bool(value: str) -> bool:
    return value.lower() in ("true", "1", "yes")


def _launch_setup(context, *args, **kwargs):
    usb_params = LaunchConfiguration("usb_cam_params_file").perform(context)
    ppocr_params = LaunchConfiguration("ppocr_params_file").perform(context)
    webviz_params = LaunchConfiguration("webviz_params_file").perform(context)
    http_port = LaunchConfiguration("http_port").perform(context)
    models_dir = LaunchConfiguration("models_dir").perform(context)
    auto_download = LaunchConfiguration("auto_download_models").perform(context)
    enable_webviz = _as_bool(LaunchConfiguration("enable_webviz").perform(context))

    usb_cam_launch = perform_substitutions(context, [
        PathJoinSubstitution([
            FindPackageShare("topsbot_usb_cam"), "launch", "usb_cam.launch.py",
        ]),
    ])
    ppocr_launch = perform_substitutions(context, [
        PathJoinSubstitution([
            FindPackageShare("topsbot_ppocr"), "launch", "ppocr_node.launch.py",
        ]),
    ])
    webviz_launch = perform_substitutions(context, [
        PathJoinSubstitution([
            FindPackageShare("topsbot_webviz"), "launch", "webviz.launch.py",
        ]),
    ])

    actions = [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(usb_cam_launch),
            launch_arguments={"params_file": usb_params}.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(ppocr_launch),
            launch_arguments={
                "params_file": ppocr_params,
                "models_dir": models_dir,
                "auto_download_models": auto_download,
            }.items(),
        ),
    ]

    if enable_webviz:
        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(webviz_launch),
                launch_arguments={
                    "params_file": webviz_params,
                    "http_port": http_port,
                }.items(),
            )
        )
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "usb_cam_params_file",
            default_value="mjpeg2nv12_640x480_zc.yaml",
            description="topsbot_usb_cam NV12 zero-copy scene YAML",
        ),
        DeclareLaunchArgument(
            "ppocr_params_file",
            default_value="ai_infer_ppocr.yaml",
            description="PPOCR ai_infer parameter YAML",
        ),
        DeclareLaunchArgument(
            "models_dir",
            default_value="~/topsbot/models",
            description="Directory containing PPOCR .nb models",
        ),
        DeclareLaunchArgument(
            "auto_download_models",
            default_value="true",
            description="Download missing PPOCR models",
        ),
        DeclareLaunchArgument(
            "enable_webviz",
            default_value="true",
            description="Start browser visualization",
        ),
        DeclareLaunchArgument(
            "webviz_params_file",
            default_value="tb_img_ppocr.yaml",
            description="topsbot_webviz OCR overlay YAML",
        ),
        DeclareLaunchArgument(
            "http_port",
            default_value="9889",
            description="WebViz static HTTP port",
        ),
        OpaqueFunction(function=_launch_setup),
    ])
