// Copyright 2026 TOPSBOT
// SPDX-License-Identifier: Apache-2.0
//
// DB text-detection post-processing, ported from the model-zoo PP-OCR sample
// (postprocess.{hpp,cpp}) so the topsbot_ppocr plugin can reuse it unchanged.

#ifndef TOPSBOT_PPOCR__POSTPROCESS_HPP_
#define TOPSBOT_PPOCR__POSTPROCESS_HPP_

#include <vector>

#include <opencv2/core.hpp>

namespace topsbot_ppocr
{

/// A detected text quadrilateral with its four corners in image pixel
/// coordinates (clockwise from top-left after OrderPointsClockwise).
struct OCRBox
{
  int x1{0};
  int y1{0};
  int x2{0};
  int y2{0};
  int x3{0};
  int y3{0};
  int x4{0};
  int y4{0};
};

using OCRBoxVec = std::vector<OCRBox>;

class PostProcessor
{
public:
  /// Find text boxes from the DB probability map (`pred`, CV_32F) and the
  /// binarized segmentation bitmap. Returns a vector of 4-point boxes in
  /// `dest_width x dest_height` (full image) pixel coordinates.
  std::vector<std::vector<std::vector<int>>> BoxesFromBitmap(
    const cv::Mat & pred,
    const cv::Mat & bitmap,
    float box_thresh,
    float unclip_ratio,
    bool use_polygon_score,
    int dest_width,
    int dest_height);

  /// Order the four points clockwise and clamp to the image bounds,
  /// dropping degenerate boxes. Returns OCRBox records.
  OCRBoxVec FilterTagDetRes(
    std::vector<std::vector<std::vector<int>>> boxes,
    const cv::Mat & input_img);

private:
  std::vector<std::vector<float>> GetMiniBoxes(cv::RotatedRect box, float & ssid);
  float PolygonScoreAcc(std::vector<cv::Point> contour, const cv::Mat & pred);
  float BoxScoreFast(std::vector<std::vector<float>> box_array, const cv::Mat & pred);
  cv::RotatedRect UnClip(std::vector<std::vector<float>> box, float unclip_ratio);
  void GetContourArea(const std::vector<std::vector<float>> & box,
    float unclip_ratio, float & distance);
  std::vector<std::vector<int>> OrderPointsClockwise(
    std::vector<std::vector<int>> pts);
  std::vector<std::vector<float>> Mat2Vector(const cv::Mat & mat);

  static bool XsortFp32(std::vector<float> a, std::vector<float> b);
  static bool XsortInt(std::vector<int> a, std::vector<int> b);

  template<class T>
  static T clamp(T x, T min, T max)
  {
    if (x > max) {return max;}
    if (x < min) {return min;}
    return x;
  }

  static int imax(int a, int b) {return a >= b ? a : b;}
  static int imin(int a, int b) {return a >= b ? b : a;}
};

}  // namespace topsbot_ppocr

#endif  // TOPSBOT_PPOCR__POSTPROCESS_HPP_