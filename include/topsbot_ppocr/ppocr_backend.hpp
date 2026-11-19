// Copyright 2026 TOPSBOT
// SPDX-License-Identifier: Apache-2.0

#ifndef TOPSBOT_PPOCR__PPOCR_BACKEND_HPP_
#define TOPSBOT_PPOCR__PPOCR_BACKEND_HPP_

#include <mutex>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <ta-runtime-api.h>

#include "topsbot_ai_infer/inference_backend.hpp"
#include "topsbot_ppocr/postprocess.hpp"

namespace topsbot_ppocr
{

/// Two-stage PP-OCR backend (DB text detection + CRNN/CTC recognition) as an
/// IInferenceBackend plugin. Preprocessing runs on the CPU (OpenCV); only the
/// two networks execute on the NPU. Both models share one ta_runtime
/// initialization via a reference-counted acquire/release.
///
/// Results are returned as a single OcrMeta in FrameMeta::user_metas; the
/// ai_infer_node forwards that to the tb_ocr_msgs/TbOcrTargets topic.
class PPOCRBackend final : public topsbot_ai_infer::IInferenceBackend
{
public:
  ~PPOCRBackend() override;

  bool initialize(
    const topsbot_ai_infer::ModelConfig & config, std::string & error) override;
  topsbot_ai_infer::InferenceResult infer(
    const topsbot_ai_infer::InferenceFrame & frame) override;
  const char * name() const noexcept override {return "ppocr";}

private:
  bool load_detector(const std::string & path, int device_id, std::string & error);
  bool load_recognizer(const std::string & path, int device_id, std::string & error);
  void shutdown();

  bool det_infer(const cv::Mat & bgr, std::vector<OCRBox> & boxes, std::string & error);
  bool rec_infer(const cv::Mat & crop, std::string & text, float & score, std::string & error);

  topsbot_ai_infer::ModelConfig config_;
  std::vector<std::string> dict_;

  // Detector.
  ta_runtime_context det_context_{0};
  std::vector<taconn_inout_attr_t> det_ins_attr_;
  std::vector<taconn_inout_attr_t> det_outs_attr_;
  std::vector<taconn_input_t> det_inputs_;
  std::vector<taconn_buffer_t> det_outputs_;
  int det_resize_w_{0};
  int det_resize_h_{0};
  bool det_outputs_created_{false};

  // Recognizer.
  ta_runtime_context rec_context_{0};
  std::vector<taconn_inout_attr_t> rec_ins_attr_;
  std::vector<taconn_inout_attr_t> rec_outs_attr_;
  std::vector<taconn_input_t> rec_inputs_;
  std::vector<taconn_buffer_t> rec_outputs_;
  bool rec_outputs_created_{false};

  bool runtime_acquired_{false};
  bool initialized_{false};
};

}  // namespace topsbot_ppocr

#endif  // TOPSBOT_PPOCR__PPOCR_BACKEND_HPP_