// Copyright 2026 TOPSBOT
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "topsbot_ai_infer/inference_backend.hpp"

namespace
{

void print_usage(const char * executable)
{
  std::cerr
    << "Usage: " << executable << " <input.jpg> <output.jpg> [options]\n"
    << "  --det-model PATH         required detector .nb model file\n"
    << "  --rec-model PATH         required recognizer .nb model file\n"
    << "  --dict-path PATH         required dictionary file\n"
    << "  --device-id INT          NPU core (default: 0)\n"
    << "  --det-db-thresh FLOAT    DB segmentation threshold (default: 0.3)\n"
    << "  --det-box-thresh FLOAT   box score threshold (default: 0.6)\n"
    << "  --rec-thresh FLOAT       recognition confidence threshold (default: 0.5)\n";
}

std::string next_argument(int argc, char ** argv, int & index)
{
  if (index + 1 >= argc) {
    return {};
  }
  return argv[++index];
}

std::string expand_user_path(const std::string & path)
{
  if (path.empty() || path.front() != '~') {
    return path;
  }
  const char * user_home = std::getenv("HOME");
  if (!user_home || user_home[0] == '\0') {
    return path;
  }
  if (path.size() == 1U || path[1] == '/') {
    return std::string(user_home) + path.substr(1);
  }
  return path;
}

std::shared_ptr<std::vector<uint8_t>> read_binary(const std::string & path)
{
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return nullptr;
  }
  return std::make_shared<std::vector<uint8_t>>(
    std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

}  // namespace

int main(int argc, char ** argv)
{
  if (argc < 3) {
    print_usage(argv[0]);
    return 1;
  }

  const std::string input_path = expand_user_path(argv[1]);
  const std::string output_path = expand_user_path(argv[2]);
  topsbot_ai_infer::ModelConfig config;
  config.backend = "ppocr";
  config.model_name = "ppocr";
  config.model_type = "ppocr";
  config.det_input_width = 480;
  config.det_input_height = 480;
  config.rec_input_width = 320;
  config.rec_input_height = 48;
  config.det_db_thresh = 0.3F;
  config.det_db_box_thresh = 0.6F;
  config.det_db_unclip_ratio = 1.5F;
  config.det_use_polygon_score = false;
  config.rec_thresh = 0.5F;
  config.use_beam_search = true;
  config.beam_size = 5;
  config.device_id = 0;

  for (int index = 3; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--det-model") {
      config.det_model_path = expand_user_path(next_argument(argc, argv, index));
    } else if (argument == "--rec-model") {
      config.rec_model_path = expand_user_path(next_argument(argc, argv, index));
    } else if (argument == "--dict-path") {
      config.dict_path = expand_user_path(next_argument(argc, argv, index));
    } else if (argument == "--device-id") {
      config.device_id = std::atoi(next_argument(argc, argv, index).c_str());
    } else if (argument == "--det-db-thresh") {
      config.det_db_thresh = static_cast<float>(std::atof(next_argument(argc, argv, index).c_str()));
    } else if (argument == "--det-box-thresh") {
      config.det_db_box_thresh = static_cast<float>(std::atof(next_argument(argc, argv, index).c_str()));
    } else if (argument == "--rec-thresh") {
      config.rec_thresh = static_cast<float>(std::atof(next_argument(argc, argv, index).c_str()));
    } else {
      std::cerr << "unknown option: " << argument << '\n';
      print_usage(argv[0]);
      return 1;
    }
  }

  if (config.det_model_path.empty() || config.rec_model_path.empty() || config.dict_path.empty()) {
    std::cerr << "error: --det-model, --rec-model, and --dict-path are required\n";
    return 1;
  }
  if (!std::filesystem::is_regular_file(input_path)) {
    std::cerr << "error: input file not found: " << input_path << '\n';
    return 1;
  }

  auto jpeg = read_binary(input_path);
  if (!jpeg || jpeg->empty()) {
    std::cerr << "error: failed to read input JPEG\n";
    return 1;
  }
  std::string error;
  auto backend = topsbot_ai_infer::create_inference_backend(config.backend, &error);
  if (!backend || !backend->initialize(config, error)) {
    std::cerr << "error: PP-OCR initialization failed: " << error << '\n';
    return 1;
  }

  topsbot_ai_infer::InferenceFrame frame;
  frame.owner = jpeg;
  frame.data = jpeg->data();
  frame.data_size = jpeg->size();
  frame.format = topsbot_ai_infer::PixelFormat::Jpeg;
  frame.sequence = 1U;
  frame.source_id = "file";
  const auto result = backend->infer(frame);
  if (!result.success) {
    std::cerr << "error: PP-OCR inference failed: " << result.error << '\n';
    return 1;
  }

  cv::Mat visualization = cv::imread(input_path, cv::IMREAD_COLOR);
  if (visualization.empty()) {
    std::cerr << "error: failed to decode visualization image\n";
    return 1;
  }

  size_t line_count = 0;
  for (const auto & meta : result.frame.user_metas) {
    if (!meta || meta->type != topsbot_ai_infer::MetaType::Ocr) {
      continue;
    }
    auto * ocr = dynamic_cast<topsbot_ai_infer::OcrMeta *>(meta.get());
    if (ocr == nullptr) {
      continue;
    }
    for (const auto & line : ocr->lines) {
      ++line_count;
      std::vector<cv::Point> points;
      for (int corner = 0; corner < 4; ++corner) {
        points.emplace_back(
          static_cast<int>(line.box[corner].x),
          static_cast<int>(line.box[corner].y));
      }
      cv::polylines(visualization, points, true, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
      // Draw only the index number on the image (cv::putText does not support CJK).
      // The full transcription (including Chinese) is printed to the console below.
      const int label_x = std::max(0, static_cast<int>(line.box[0].x));
      const int label_y = std::max(15, static_cast<int>(line.box[0].y) - 5);
      std::ostringstream label;
      label << "#" << line_count;
      const double font_scale = std::max(0.4, visualization.cols / 1600.0);
      cv::putText(
        visualization, label.str(), cv::Point(label_x, label_y),
        cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
      cv::putText(
        visualization, label.str(), cv::Point(label_x, label_y),
        cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
      std::cout << line_count << ". [" << std::fixed << std::setprecision(4)
                << line.score << "] " << line.transcription << "\n";
    }
  }

  const auto parent = std::filesystem::path(output_path).parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent);
  }
  if (!cv::imwrite(output_path, visualization)) {
    std::cerr << "error: failed to write output image: " << output_path << '\n';
    return 1;
  }
  std::cout << "Detected " << line_count << " text lines\n";
  std::cout << "Timing ms: input_copy=" << result.timing.input_copy_ms
            << " preprocess=" << result.timing.preprocess_ms
            << " infer=" << result.timing.infer_ms
            << " post=" << result.timing.postprocess_ms << '\n'
            << "Wrote " << output_path << '\n';
  return 0;
}
