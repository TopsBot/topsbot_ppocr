// Copyright 2026 TOPSBOT
// SPDX-License-Identifier: Apache-2.0

#include "topsbot_ppocr/ppocr_backend.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <mutex>
#include <numeric>
#include <utility>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <pluginlib/class_list_macros.hpp>

#include "topsbot_ppocr/ppocr_utils.hpp"

namespace topsbot_ppocr
{

using topsbot_ai_infer::BaseMeta;
using topsbot_ai_infer::InferenceFrame;
using topsbot_ai_infer::InferenceResult;
using topsbot_ai_infer::ModelConfig;
using topsbot_ai_infer::OcrMeta;
using topsbot_ai_infer::PixelFormat;

namespace
{

using SteadyClock = std::chrono::steady_clock;

float elapsed_ms(const SteadyClock::time_point & begin, const SteadyClock::time_point & end)
{
  return std::chrono::duration<float, std::milli>(end - begin).count();
}

std::mutex runtime_mutex;
size_t runtime_users = 0U;

bool acquire_runtime()
{
  std::lock_guard<std::mutex> lock(runtime_mutex);
  if (runtime_users == 0U && ta_runtime_init() != NNRT_SUCCESS) {
    return false;
  }
  ++runtime_users;
  return true;
}

void release_runtime()
{
  std::lock_guard<std::mutex> lock(runtime_mutex);
  if (runtime_users == 0U) {
    return;
  }
  --runtime_users;
  if (runtime_users == 0U) {
    (void)ta_runtime_deinit();
  }
}

/// Allocate an aligned CPU input buffer and register it with the runtime via
/// ta_runtime_set_input_cva. Returns false and sets `error` on failure.
bool setup_inputs(
  const ta_runtime_context & context,
  const std::vector<taconn_inout_attr_t> & attrs,
  std::vector<taconn_input_t> & inputs,
  std::string & error)
{
  for (auto & in : inputs) {
    if (in.data != nullptr) {
      std::free(in.data);
    }
  }
  inputs.clear();
  inputs.resize(attrs.size());
  for (size_t i = 0; i < attrs.size(); ++i) {
    const size_t bytes = tensor_byte_size(attrs[i]);
    if (bytes == 0U || bytes > std::numeric_limits<uint32_t>::max()) {
      error = "unsupported PP-OCR input tensor format";
      return false;
    }
    inputs[i].index = static_cast<uint32_t>(i);
    inputs[i].size = static_cast<uint32_t>(bytes);
    void * data = nullptr;
    if (posix_memalign(&data, 256, bytes) != 0 || data == nullptr) {
      error = "failed to allocate PP-OCR input buffer";
      return false;
    }
    std::memset(data, 0, bytes);
    inputs[i].data = data;
  }
  if (ta_runtime_set_input_cva(
      const_cast<ta_runtime_context *>(&context),
      static_cast<uint32_t>(inputs.size()), inputs.data()) != NNRT_SUCCESS)
  {
    error = "ta_runtime_set_input_cva failed";
    return false;
  }
  return true;
}

bool setup_outputs(
  const ta_runtime_context & context,
  const std::vector<taconn_inout_attr_t> & attrs,
  std::vector<taconn_buffer_t> & outputs,
  bool & outputs_created,
  std::string & error)
{
  outputs.clear();
  outputs.resize(attrs.size());
  outputs_created = false;
  for (size_t i = 0; i < attrs.size(); ++i) {
    const size_t bytes = tensor_byte_size(attrs[i]);
    if (bytes == 0U) {
      error = "unsupported PP-OCR output tensor format";
      return false;
    }
    outputs[i] = {};
    if (ta_runtime_create_buffer(
        const_cast<ta_runtime_context *>(&context),
        static_cast<uint32_t>(bytes), &outputs[i]) != NNRT_SUCCESS)
    {
      error = "ta_runtime_create_buffer failed";
      return false;
    }
    outputs_created = true;
  }
  if (ta_runtime_set_output(
      const_cast<ta_runtime_context *>(&context),
      static_cast<uint32_t>(outputs.size()), outputs.data()) != NNRT_SUCCESS)
  {
    error = "ta_runtime_set_output failed";
    return false;
  }
  return true;
}

void destroy_outputs(
  const ta_runtime_context & context,
  std::vector<taconn_buffer_t> & outputs,
  bool & outputs_created)
{
  if (outputs_created && context != 0) {
    for (auto & buf : outputs) {
      if (buf.data != nullptr) {
        (void)ta_runtime_destroy_buffer(const_cast<ta_runtime_context *>(&context), &buf);
      }
    }
  }
  outputs.clear();
  outputs_created = false;
}

void free_inputs(std::vector<taconn_input_t> & inputs)
{
  for (auto & in : inputs) {
    std::free(in.data);
  }
  inputs.clear();
}

/// Perspective-warp the text quadrilateral into an upright strip, matching the
/// model-zoo getRotateCropImage semantics (min side 16, rotate 90 if needed).
cv::Mat rotate_crop_image(const cv::Mat & img, const OCRBox & box)
{
  std::vector<cv::Point2f> points{
    cv::Point2f(static_cast<float>(box.x1), static_cast<float>(box.y1)),
    cv::Point2f(static_cast<float>(box.x2), static_cast<float>(box.y2)),
    cv::Point2f(static_cast<float>(box.x3), static_cast<float>(box.y3)),
    cv::Point2f(static_cast<float>(box.x4), static_cast<float>(box.y4))};

  auto distance = [](const cv::Point2f & a, const cv::Point2f & b) {
    return std::sqrt(
      (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
  };

  const double width1 = distance(points[0], points[1]);
  const double width2 = distance(points[2], points[3]);
  int crop_width = static_cast<int>(std::max(width1, width2));
  const double height1 = distance(points[0], points[3]);
  const double height2 = distance(points[1], points[2]);
  int crop_height = static_cast<int>(std::max(height1, height2));

  crop_width = std::max(16, crop_width);
  crop_height = std::max(16, crop_height);

  std::vector<cv::Point2f> pts_std{
    cv::Point2f(0.0F, 0.0F),
    cv::Point2f(static_cast<float>(crop_width), 0.0F),
    cv::Point2f(static_cast<float>(crop_width), static_cast<float>(crop_height)),
    cv::Point2f(0.0F, static_cast<float>(crop_height))};

  const cv::Mat M = cv::getPerspectiveTransform(points, pts_std);
  cv::Mat dst_img;
  cv::warpPerspective(
    img, dst_img, M, cv::Size(crop_width, crop_height),
    cv::INTER_CUBIC, cv::BORDER_REPLICATE);

  if (static_cast<double>(dst_img.rows) / dst_img.cols >= 1.5) {
    cv::rotate(dst_img, dst_img, cv::ROTATE_90_COUNTERCLOCKWISE);
  }
  return dst_img;
}

}  // namespace

PPOCRBackend::~PPOCRBackend()
{
  shutdown();
}

bool PPOCRBackend::initialize(const ModelConfig & config, std::string & error)
{
  shutdown();
  error.clear();
  config_ = config;

  if (config_.det_model_path.empty() ||
    !std::filesystem::is_regular_file(config_.det_model_path))
  {
    error = "PP-OCR det model not found: " + config_.det_model_path;
    return false;
  }
  if (config_.rec_model_path.empty() ||
    !std::filesystem::is_regular_file(config_.rec_model_path))
  {
    error = "PP-OCR rec model not found: " + config_.rec_model_path;
    return false;
  }
  if (config_.dict_path.empty()) {
    error = "ppocr.dict_path is required";
    return false;
  }
  if (config_.beam_size < 1 || config_.beam_size > 40) {
    error = "ppocr.beam_size must be in [1, 40]";
    return false;
  }
  if (!read_dict(config_.dict_path, dict_, error)) {
    return false;
  }

  if (!acquire_runtime()) {
    error = "ta_runtime_init failed";
    return false;
  }
  runtime_acquired_ = true;

  if (!load_detector(config_.det_model_path, config_.device_id, error) ||
    !load_recognizer(config_.rec_model_path, config_.device_id, error))
  {
    shutdown();
    return false;
  }

  initialized_ = true;
  return true;
}

bool PPOCRBackend::load_detector(
  const std::string & path, const int device_id, std::string & error)
{
  det_resize_w_ = 0;
  det_resize_h_ = 0;
  if (ta_runtime_load_model_from_file(
      &det_context_, path.c_str(), static_cast<uint32_t>(device_id)) != NNRT_SUCCESS)
  {
    error = "failed to load PP-OCR det model on NPU " + std::to_string(device_id);
    return false;
  }

  taconn_input_output_num_t counts{};
  if (ta_runtime_query(&det_context_, TACONN_QUERY_IN_OUT_NUM, &counts) != NNRT_SUCCESS ||
    counts.input_num < 1U || counts.output_num < 1U)
  {
    error = "PP-OCR det model requires at least one input and one output";
    return false;
  }

  det_ins_attr_.resize(counts.input_num);
  det_outs_attr_.resize(counts.output_num);
  for (uint32_t i = 0; i < counts.input_num; ++i) {
    det_ins_attr_[i] = {};
    det_ins_attr_[i].index = i;
    if (ta_runtime_query(&det_context_, TACONN_QUERY_INPUT_ATTR, &det_ins_attr_[i]) !=
      NNRT_SUCCESS)
    {
      error = "failed to query PP-OCR det input attributes";
      return false;
    }
  }
  for (uint32_t i = 0; i < counts.output_num; ++i) {
    det_outs_attr_[i] = {};
    det_outs_attr_[i].index = i;
    if (ta_runtime_query(&det_context_, TACONN_QUERY_OUTPUT_ATTR, &det_outs_attr_[i]) !=
      NNRT_SUCCESS)
    {
      error = "failed to query PP-OCR det output attributes";
      return false;
    }
  }

  if (!setup_inputs(det_context_, det_ins_attr_, det_inputs_, error) ||
    !setup_outputs(det_context_, det_outs_attr_, det_outputs_, det_outputs_created_, error))
  {
    return false;
  }
  return true;
}

bool PPOCRBackend::load_recognizer(
  const std::string & path, const int device_id, std::string & error)
{
  if (ta_runtime_load_model_from_file(
      &rec_context_, path.c_str(), static_cast<uint32_t>(device_id)) != NNRT_SUCCESS)
  {
    error = "failed to load PP-OCR rec model on NPU " + std::to_string(device_id);
    return false;
  }

  taconn_input_output_num_t counts{};
  if (ta_runtime_query(&rec_context_, TACONN_QUERY_IN_OUT_NUM, &counts) != NNRT_SUCCESS ||
    counts.input_num < 1U || counts.output_num < 1U)
  {
    error = "PP-OCR rec model requires at least one input and one output";
    return false;
  }

  rec_ins_attr_.resize(counts.input_num);
  rec_outs_attr_.resize(counts.output_num);
  for (uint32_t i = 0; i < counts.input_num; ++i) {
    rec_ins_attr_[i] = {};
    rec_ins_attr_[i].index = i;
    if (ta_runtime_query(&rec_context_, TACONN_QUERY_INPUT_ATTR, &rec_ins_attr_[i]) !=
      NNRT_SUCCESS)
    {
      error = "failed to query PP-OCR rec input attributes";
      return false;
    }
  }
  for (uint32_t i = 0; i < counts.output_num; ++i) {
    rec_outs_attr_[i] = {};
    rec_outs_attr_[i].index = i;
    if (ta_runtime_query(&rec_context_, TACONN_QUERY_OUTPUT_ATTR, &rec_outs_attr_[i]) !=
      NNRT_SUCCESS)
    {
      error = "failed to query PP-OCR rec output attributes";
      return false;
    }
  }

  if (!setup_inputs(rec_context_, rec_ins_attr_, rec_inputs_, error) ||
    !setup_outputs(rec_context_, rec_outs_attr_, rec_outputs_, rec_outputs_created_, error))
  {
    return false;
  }
  return true;
}

void PPOCRBackend::shutdown()
{
  initialized_ = false;
  dict_.clear();
  det_resize_w_ = 0;
  det_resize_h_ = 0;
  destroy_outputs(det_context_, det_outputs_, det_outputs_created_);
  destroy_outputs(rec_context_, rec_outputs_, rec_outputs_created_);
  free_inputs(det_inputs_);
  free_inputs(rec_inputs_);
  det_ins_attr_.clear();
  det_outs_attr_.clear();
  rec_ins_attr_.clear();
  rec_outs_attr_.clear();
  if (det_context_ != 0) {
    (void)ta_runtime_destroy_context(&det_context_);
    det_context_ = 0;
  }
  if (rec_context_ != 0) {
    (void)ta_runtime_destroy_context(&rec_context_);
    rec_context_ = 0;
  }
  if (runtime_acquired_) {
    release_runtime();
    runtime_acquired_ = false;
  }
}

bool PPOCRBackend::det_infer(
  const cv::Mat & bgr, std::vector<OCRBox> & boxes, std::string & error)
{
  const int in_w = config_.det_input_width;
  const int in_h = config_.det_input_height;
  if (bgr.empty()) {
    error = "empty input for det";
    return false;
  }

  // --- CPU preprocess (ratio resize + BGR->RGB + mean/std + pad) ---
  const int src_h = bgr.rows;
  const int src_w = bgr.cols;
  const int size_max = std::max(src_h, src_w);
  const float ratio = size_max >= in_h
    ? static_cast<float>(in_h) / static_cast<float>(size_max) : 1.0F;
  const int resize_h = static_cast<int>(src_h * ratio);
  const int resize_w = static_cast<int>(src_w * ratio);

  cv::Mat resized;
  if (src_h != resize_h || src_w != resize_w) {
    cv::resize(bgr, resized, cv::Size(resize_w, resize_h));
  } else {
    resized = bgr.clone();
  }
  cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

  cv::Mat bordered;
  if (resize_w < in_w || resize_h < in_h) {
    cv::copyMakeBorder(
      resized, bordered, 0, in_h - resize_h, 0, in_w - resize_w,
      cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
  } else {
    bordered = resized.clone();
  }

  cv::Mat dst_float;
  bordered.convertTo(dst_float, CV_32FC3);
  const float det_mean[3] = {123.675F, 116.28F, 103.53F};
  const float det_std[3] = {0.01712471615720524F, 0.017506964285714285F,
    0.017429155555555555F};
  std::vector<cv::Mat> channels;
  cv::split(dst_float, channels);
  for (int c = 0; c < 3; ++c) {
    channels[c] = (channels[c] - det_mean[c]) * det_std[c];
  }
  cv::merge(channels, dst_float);

  const size_t total = static_cast<size_t>(in_w) * in_h * 3U;
  std::vector<float> tensor(total, 0.0F);
  const size_t plane = static_cast<size_t>(in_w) * in_h;
  std::vector<cv::Mat> split_norm;
  cv::split(dst_float, split_norm);
  for (int c = 0; c < 3; ++c) {
    std::memcpy(tensor.data() + c * plane, split_norm[c].data, plane * sizeof(float));
  }
  quantize_input(tensor.data(), det_inputs_[0].data, total, det_ins_attr_[0]);

  // --- infer ---
  if (ta_runtime_run_network(&det_context_) != NNRT_SUCCESS ||
    ta_runtime_invalidate_buffer(&det_context_, det_outputs_.data()) != NNRT_SUCCESS)
  {
    error = "PP-OCR det NPU inference failed";
    return false;
  }

  // --- postprocess ---
  std::vector<float> pred;
  dequantize_output(det_outputs_[0].data, det_outs_attr_[0], pred);
  const size_t exp = static_cast<size_t>(in_w) * in_h;
  if (pred.size() < exp) {
    error = "PP-OCR det output smaller than expected";
    return false;
  }

  std::vector<uint8_t> cbuf(exp);
  for (size_t i = 0; i < exp; ++i) {
    cbuf[i] = static_cast<uint8_t>(
      std::clamp((pred[i] >= 0.0F ? pred[i] : 0.0F), 0.0F, 1.0F) *
      255.0F);
  }
  // pred map is in sigmoid range [0,1]; clamp for visualization thresholding.
  cv::Mat pred_map(in_h, in_w, CV_32F, pred.data());
  cv::Mat cbuf_map(in_h, in_w, CV_8UC1, cbuf.data());

  cv::Rect valid_region(0, 0, resize_w, resize_h);
  cv::Mat pred_valid = pred_map(valid_region);
  cv::Mat cbuf_valid = cbuf_map(valid_region);

  cv::Mat bit_map;
  cv::threshold(
    cbuf_valid, bit_map, config_.det_db_thresh * 255.0F, 255.0F, cv::THRESH_BINARY);

  PostProcessor post;
  auto raw_boxes = post.BoxesFromBitmap(
    pred_valid, bit_map, config_.det_db_box_thresh, config_.det_db_unclip_ratio,
    config_.det_use_polygon_score, src_w, src_h);
  boxes = post.FilterTagDetRes(std::move(raw_boxes), bgr);
  det_resize_w_ = resize_w;
  det_resize_h_ = resize_h;
  return true;
}

bool PPOCRBackend::rec_infer(
  const cv::Mat & crop, std::string & text, float & score, std::string & error)
{
  if (crop.empty()) {
    error = "empty recognition crop";
    return false;
  }
  const int in_w = config_.rec_input_width;
  const int in_h = config_.rec_input_height;

  // --- CPU preprocess ---
  const int src_h = crop.rows;
  const int src_w = crop.cols;
  const float ratio = static_cast<float>(src_w) / static_cast<float>(src_h);
  const float img_ratio = static_cast<float>(in_w) / static_cast<float>(in_h);
  int resized_w = 0;
  int resized_h = 0;
  int padding_w = 0;
  if (ratio > img_ratio) {
    resized_w = in_w;
    resized_h = in_h;
  } else {
    resized_h = in_h;
    resized_w = static_cast<int>(static_cast<float>(in_h) * ratio);
    padding_w = static_cast<int>(static_cast<float>(in_h) * img_ratio);
  }

  cv::Mat resized;
  if (src_h != resized_h || src_w != resized_w) {
    cv::resize(crop, resized, cv::Size(resized_w, resized_h));
  } else {
    resized = crop.clone();
  }
  cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

  cv::Mat dst_float;
  resized.convertTo(dst_float, CV_32FC3);
  const float rec_mean[3] = {127.5F, 127.5F, 127.5F};
  const float rec_std[3] = {0.007874016F, 0.007874016F, 0.007874016F};
  std::vector<cv::Mat> channels;
  cv::split(dst_float, channels);
  for (int c = 0; c < 3; ++c) {
    channels[c] = (channels[c] - rec_mean[c]) * rec_std[c];
  }
  cv::merge(channels, dst_float);

  cv::Mat output_image;
  if (resized_w < padding_w) {
    const int right = padding_w - resized_w;
    const int bottom = in_h - resized_h;
    cv::copyMakeBorder(
      dst_float, output_image, 0, bottom, 0, right,
      cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
  } else {
    output_image = dst_float.clone();
  }

  const size_t total = static_cast<size_t>(in_w) * in_h * 3U;
  const size_t plane = static_cast<size_t>(in_w) * in_h;
  std::vector<float> tensor(total, 0.0F);
  std::vector<cv::Mat> split_norm;
  cv::split(output_image, split_norm);
  for (int c = 0; c < 3; ++c) {
    std::memcpy(tensor.data() + c * plane, split_norm[c].data, plane * sizeof(float));
  }
  quantize_input(tensor.data(), rec_inputs_[0].data, total, rec_ins_attr_[0]);

  // --- infer ---
  if (ta_runtime_run_network(&rec_context_) != NNRT_SUCCESS ||
    ta_runtime_invalidate_buffer(&rec_context_, rec_outputs_.data()) != NNRT_SUCCESS)
  {
    error = "PP-OCR rec NPU inference failed";
    return false;
  }

  // --- decode (CTC greedy / beam search) ---
  std::vector<float> probs;
  dequantize_output(rec_outputs_[0].data, rec_outs_attr_[0], probs);

  const int num_classes = static_cast<int>(dict_.size());
  if (num_classes <= 0) {
    error = "empty recognition dictionary";
    return false;
  }
  int max_seq_len = 40;
  if (probs.size() % static_cast<size_t>(num_classes) == 0U) {
    max_seq_len = static_cast<int>(probs.size() / static_cast<size_t>(num_classes));
  }

  struct Beam
  {
    std::vector<int> prefix;
    double seg_score{1.0};
    std::vector<float> confs;
  };

  text.clear();
  std::vector<float> confs;
  const int beam_width = std::max(1, config_.beam_size);

  if (config_.use_beam_search) {
    std::vector<Beam> beams;
    beams.push_back(Beam{});
    for (int t = 0; t < max_seq_len; ++t) {
      std::vector<Beam> next;
      next.reserve(static_cast<size_t>(beams.size()) * beam_width);
      const float * step = probs.data() + static_cast<size_t>(t) * num_classes;
      for (const auto & beam : beams) {
        std::vector<int> order(num_classes);
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(
          order.begin(), order.begin() + beam_width, order.end(),
          [step](int a, int b) {return step[a] > step[b];});
        for (int k = 0; k < beam_width; ++k) {
          const int c = order[k];
          Beam nb = beam;
          nb.prefix.push_back(c);
          nb.seg_score *= std::max(step[c], 1e-40F);
          nb.confs.push_back(step[c]);
          next.push_back(std::move(nb));
        }
      }
      std::sort(next.begin(), next.end(),
        [](const Beam & a, const Beam & b) {return a.seg_score > b.seg_score;});
      if (next.size() > static_cast<size_t>(beam_width)) {
        next.resize(beam_width);
      }
      beams = std::move(next);
    }
    auto best = std::max_element(beams.begin(), beams.end(),
      [](const Beam & a, const Beam & b) {return a.seg_score < b.seg_score;});
    if (best == beams.end()) {
      text = "###";
    } else {
      int pre = -1;
      for (size_t i = 0; i < best->prefix.size(); ++i) {
        const int c = best->prefix[i];
        if (c == 0) {pre = c; continue;}
        if (c == pre) {continue;}
        text += dict_[static_cast<size_t>(c)];
        confs.push_back(best->confs[i]);
        pre = c;
      }
    }
  } else {
    int pre = -1;
    for (int t = 0; t < max_seq_len; ++t) {
      const float * step = probs.data() + static_cast<size_t>(t) * num_classes;
      const int c = static_cast<int>(
        std::max_element(step, step + num_classes) - step);
      if (c == 0) {pre = c; continue;}
      if (c == pre) {continue;}
      text += dict_[static_cast<size_t>(c)];
      confs.push_back(step[c]);
      pre = c;
    }
  }

  if (confs.empty()) {
    score = 0.0F;
    text = "###";
  } else {
    score = std::accumulate(confs.begin(), confs.end(), 0.0F) /
      static_cast<float>(confs.size());
    if (std::isnan(score)) {
      score = 0.0F;
      text = "###";
    }
  }
  return true;
}

InferenceResult PPOCRBackend::infer(const InferenceFrame & frame)
{
  InferenceResult result;
  result.backend = name();
  result.model_name = config_.model_name;
  result.device_id = config_.device_id;
  result.frame.source_id = frame.source_id;
  result.frame.frame_number = frame.sequence;
  result.frame.timestamp_ns = frame.timestamp_ns;
  result.frame.tags["backend"] = name();
  result.frame.tags["model"] = config_.model_name;
  result.frame.tags["device_id"] = std::to_string(config_.device_id);

  if (!initialized_) {
    result.error = "ppocr backend is not initialized";
    return result;
  }

  // --- Decode the incoming frame to BGR on the CPU ---
  const auto decode_begin = SteadyClock::now();
  cv::Mat bgr;
  if (frame.format == PixelFormat::Nv12) {
    const size_t expected = static_cast<size_t>(frame.width) *
      static_cast<size_t>(frame.height) * 3U / 2U;
    if (!frame.data || frame.width <= 0 || frame.height <= 0 ||
      frame.data_size < expected)
    {
      result.error = "invalid NV12 input frame";
      return result;
    }
    cv::Mat nv12(frame.height * 3 / 2, frame.width, CV_8UC1,
      const_cast<uint8_t *>(frame.data));
    cv::cvtColor(nv12, bgr, cv::COLOR_YUV2BGR_NV12);
  } else if (frame.format == PixelFormat::Jpeg) {
    if (!frame.data || frame.data_size == 0U) {
      result.error = "invalid JPEG input";
      return result;
    }
    bgr = cv::imdecode(
      cv::Mat(1, static_cast<int>(frame.data_size), CV_8UC1,
        const_cast<uint8_t *>(frame.data)),
      cv::IMREAD_COLOR);
    if (bgr.empty()) {
      result.error = "failed to decode JPEG input";
      return result;
    }
  } else {
    const int step = frame.step > 0 ? frame.step : frame.width * 3;
    const size_t expected = static_cast<size_t>(step) * frame.height;
    if (!frame.data || frame.width <= 0 || frame.height <= 0 ||
      step < frame.width * 3 || frame.data_size < expected)
    {
      result.error = "invalid BGR8 input frame";
      return result;
    }
    bgr = cv::Mat(frame.height, frame.width, CV_8UC3,
      const_cast<uint8_t *>(frame.data), step).clone();
  }
  const auto decode_end = SteadyClock::now();
  result.timing.input_copy_ms = elapsed_ms(decode_begin, decode_end);

  // --- Det ---
  const auto det_begin = SteadyClock::now();
  std::vector<OCRBox> boxes;
  if (!det_infer(bgr, boxes, result.error)) {
    return result;
  }
  const auto det_end = SteadyClock::now();
  const float det_total_ms = elapsed_ms(det_begin, det_end);

  // --- Sort boxes in reading order (top-to-bottom, left-to-right) ---
  std::sort(boxes.begin(), boxes.end(),
    [](const OCRBox & a, const OCRBox & b) {
      // Use top-left corner for sorting
      const float ay = (a.y1 + a.y2) * 0.5F;  // average y of top edge
      const float by = (b.y1 + b.y2) * 0.5F;
      const float ax = (a.x1 + a.x4) * 0.5F;  // average x of left edge
      const float bx = (b.x1 + b.x4) * 0.5F;

      // First sort by vertical position (top to bottom)
      // Allow some tolerance for boxes on the same line
      const float row_tolerance = 10.0F;
      if (std::abs(ay - by) > row_tolerance) {
        return ay < by;
      }
      // Within the same row, sort by horizontal position (left to right)
      return ax < bx;
    });

  // --- Rec (one inference per detected box) ---
  auto meta = std::make_shared<OcrMeta>();
  meta->model_name = config_.model_name;
  const float rec_thresh = config_.rec_thresh;

  const auto rec_begin = SteadyClock::now();
  for (const auto & box : boxes) {
    const cv::Mat crop = rotate_crop_image(bgr, box);
    std::string text;
    float score = 0.0F;
    if (!rec_infer(crop, text, score, result.error)) {
      return result;
    }
    if (text == "###" || score < rec_thresh) {
      continue;
    }
    OcrMeta::OcrLine line;
    line.box[0] = {static_cast<float>(box.x1), static_cast<float>(box.y1), score};
    line.box[1] = {static_cast<float>(box.x2), static_cast<float>(box.y2), score};
    line.box[2] = {static_cast<float>(box.x3), static_cast<float>(box.y3), score};
    line.box[3] = {static_cast<float>(box.x4), static_cast<float>(box.y4), score};
    line.transcription = std::move(text);
    line.score = score;
    meta->lines.push_back(std::move(line));
  }
  const auto rec_end = SteadyClock::now();
  float rec_total_ms = elapsed_ms(rec_begin, rec_end);

  result.frame.user_metas.push_back(std::move(meta));
  result.timing.preprocess_ms = det_total_ms * 0.5F;  // det dominates preprocess
  result.timing.infer_ms = det_total_ms * 0.5F + rec_total_ms;
  result.timing.postprocess_ms = 0.0F;
  result.success = true;
  return result;
}

}  // namespace topsbot_ppocr

PLUGINLIB_EXPORT_CLASS(
  topsbot_ppocr::PPOCRBackend, topsbot_ai_infer::IInferenceBackend)