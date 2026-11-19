// Copyright 2026 TOPSBOT
// SPDX-License-Identifier: Apache-2.0
//
// Port of the model-zoo PP-OCR DB post-processing (postprocess.cpp). Algorithm
// kept byte-for-byte compatible; only the namespace/includes differ.

#include "topsbot_ppocr/postprocess.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "opencv2/imgproc.hpp"

#include "topsbot_ppocr/clipper.h"

namespace topsbot_ppocr
{

void PostProcessor::GetContourArea(const std::vector<std::vector<float>> & box,
  const float unclip_ratio, float & distance)
{
  const int pts_num = 4;
  float area = 0.0F;
  float dist = 0.0F;
  for (int i = 0; i < pts_num; i++) {
    area += box[i][0] * box[(i + 1) % pts_num][1] -
            box[i][1] * box[(i + 1) % pts_num][0];
    dist += std::sqrt(
      (box[i][0] - box[(i + 1) % pts_num][0]) *
        (box[i][0] - box[(i + 1) % pts_num][0]) +
      (box[i][1] - box[(i + 1) % pts_num][1]) *
        (box[i][1] - box[(i + 1) % pts_num][1]));
  }
  area = std::fabs(area / 2.0F);
  distance = area * unclip_ratio / dist;
}

cv::RotatedRect PostProcessor::UnClip(std::vector<std::vector<float>> box,
  float unclip_ratio)
{
  float distance = 1.0F;
  GetContourArea(box, unclip_ratio, distance);

  ClipperLib::ClipperOffset offset;
  ClipperLib::Path p;
  p << ClipperLib::IntPoint(static_cast<ClipperLib::cInt>(box[0][0]),
        static_cast<ClipperLib::cInt>(box[0][1]))
    << ClipperLib::IntPoint(static_cast<ClipperLib::cInt>(box[1][0]),
        static_cast<ClipperLib::cInt>(box[1][1]))
    << ClipperLib::IntPoint(static_cast<ClipperLib::cInt>(box[2][0]),
        static_cast<ClipperLib::cInt>(box[2][1]))
    << ClipperLib::IntPoint(static_cast<ClipperLib::cInt>(box[3][0]),
        static_cast<ClipperLib::cInt>(box[3][1]));
  offset.AddPath(p, ClipperLib::jtRound, ClipperLib::etClosedPolygon);

  ClipperLib::Paths soln;
  offset.Execute(soln, distance);

  std::vector<cv::Point2f> points;
  for (std::size_t j = 0; j < soln.size(); j++) {
    for (std::size_t i = 0; i < soln[soln.size() - 1].size(); i++) {
      points.emplace_back(
        static_cast<float>(soln[j][i].X), static_cast<float>(soln[j][i].Y));
    }
  }

  cv::RotatedRect res;
  if (points.empty()) {
    res = cv::RotatedRect(cv::Point2f(0.0F, 0.0F), cv::Size2f(1.0F, 1.0F), 0.0F);
  } else {
    res = cv::minAreaRect(points);
  }
  return res;
}

float PostProcessor::BoxScoreFast(std::vector<std::vector<float>> box_array,
  const cv::Mat & pred)
{
  const auto array = box_array;
  const int width = pred.cols;
  const int height = pred.rows;

  float box_x[4] = {array[0][0], array[1][0], array[2][0], array[3][0]};
  float box_y[4] = {array[0][1], array[1][1], array[2][1], array[3][1]};

  const int xmin = clamp(
    static_cast<int>(std::floor(*std::min_element(box_x, box_x + 4))), 0, width - 1);
  const int xmax = clamp(
    static_cast<int>(std::ceil(*std::max_element(box_x, box_x + 4))), 0, width - 1);
  const int ymin = clamp(
    static_cast<int>(std::floor(*std::min_element(box_y, box_y + 4))), 0, height - 1);
  const int ymax = clamp(
    static_cast<int>(std::ceil(*std::max_element(box_y, box_y + 4))), 0, height - 1);

  cv::Mat mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8UC1);

  cv::Point root_point[4];
  root_point[0] = cv::Point(
    static_cast<int>(array[0][0]) - xmin, static_cast<int>(array[0][1]) - ymin);
  root_point[1] = cv::Point(
    static_cast<int>(array[1][0]) - xmin, static_cast<int>(array[1][1]) - ymin);
  root_point[2] = cv::Point(
    static_cast<int>(array[2][0]) - xmin, static_cast<int>(array[2][1]) - ymin);
  root_point[3] = cv::Point(
    static_cast<int>(array[3][0]) - xmin, static_cast<int>(array[3][1]) - ymin);
  const cv::Point * ppt[1] = {root_point};
  int npt[] = {4};
  cv::fillPoly(mask, ppt, npt, 1, cv::Scalar(1));

  cv::Mat cropped_img;
  pred(cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1)).copyTo(cropped_img);
  return static_cast<float>(cv::mean(cropped_img, mask)[0]);
}

float PostProcessor::PolygonScoreAcc(std::vector<cv::Point> contour,
  const cv::Mat & pred)
{
  const int width = pred.cols;
  const int height = pred.rows;
  std::vector<float> box_x;
  std::vector<float> box_y;
  for (auto & pt : contour) {
    box_x.push_back(static_cast<float>(pt.x));
    box_y.push_back(static_cast<float>(pt.y));
  }

  const int xmin = clamp(
    static_cast<int>(std::floor(*std::min_element(box_x.begin(), box_x.end()))),
    0, width - 1);
  const int xmax = clamp(
    static_cast<int>(std::ceil(*std::max_element(box_x.begin(), box_x.end()))),
    0, width - 1);
  const int ymin = clamp(
    static_cast<int>(std::floor(*std::min_element(box_y.begin(), box_y.end()))),
    0, height - 1);
  const int ymax = clamp(
    static_cast<int>(std::ceil(*std::max_element(box_y.begin(), box_y.end()))),
    0, height - 1);

  cv::Mat mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8UC1);

  std::vector<cv::Point> root_point(contour.size());
  for (std::size_t i = 0; i < contour.size(); ++i) {
    root_point[i] = cv::Point(
      static_cast<int>(box_x[i]) - xmin, static_cast<int>(box_y[i]) - ymin);
  }
  const cv::Point * ppt[1] = {root_point.data()};
  int npt[] = {static_cast<int>(contour.size())};
  cv::fillPoly(mask, ppt, npt, 1, cv::Scalar(1));

  cv::Mat cropped_img;
  pred(cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1)).copyTo(cropped_img);
  return static_cast<float>(cv::mean(cropped_img, mask)[0]);
}

bool PostProcessor::XsortFp32(std::vector<float> a, std::vector<float> b)
{
  if (a[0] != b[0]) {return a[0] < b[0];}
  return false;
}

bool PostProcessor::XsortInt(std::vector<int> a, std::vector<int> b)
{
  if (a[0] != b[0]) {return a[0] < b[0];}
  return false;
}

std::vector<std::vector<float>> PostProcessor::Mat2Vector(const cv::Mat & mat)
{
  std::vector<std::vector<float>> img_vec;
  for (int i = 0; i < mat.rows; ++i) {
    std::vector<float> tmp;
    tmp.reserve(mat.cols);
    for (int j = 0; j < mat.cols; ++j) {
      tmp.push_back(mat.at<float>(i, j));
    }
    img_vec.push_back(tmp);
  }
  return img_vec;
}

std::vector<std::vector<float>> PostProcessor::GetMiniBoxes(cv::RotatedRect box,
  float & ssid)
{
  ssid = std::min(box.size.width, box.size.height);
  cv::Mat points;
  cv::boxPoints(box, points);

  auto array = Mat2Vector(points);
  std::sort(array.begin(), array.end(), XsortFp32);

  std::vector<float> idx1 = array[0];
  std::vector<float> idx2 = array[1];
  std::vector<float> idx3 = array[2];
  std::vector<float> idx4 = array[3];
  if (array[3][1] <= array[2][1]) {
    idx2 = array[3];
    idx3 = array[2];
  } else {
    idx2 = array[2];
    idx3 = array[3];
  }
  if (array[1][1] <= array[0][1]) {
    idx1 = array[1];
    idx4 = array[0];
  } else {
    idx1 = array[0];
    idx4 = array[1];
  }
  array[0] = idx1;
  array[1] = idx2;
  array[2] = idx3;
  array[3] = idx4;
  return array;
}

std::vector<std::vector<std::vector<int>>> PostProcessor::BoxesFromBitmap(
  const cv::Mat & pred,
  const cv::Mat & bitmap,
  float box_thresh,
  float det_db_unclip_ratio,
  bool use_polygon_score,
  int dest_width,
  int dest_height)
{
  const int min_size = 3;
  const int max_candidates = 1000;

  const int width = bitmap.cols;
  const int height = bitmap.rows;

  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(bitmap, contours, hierarchy, cv::RETR_LIST,
    cv::CHAIN_APPROX_SIMPLE);

  const int num_contours =
    static_cast<int>(contours.size()) >= max_candidates
    ? max_candidates : static_cast<int>(contours.size());
  std::vector<std::vector<std::vector<int>>> boxes;

  for (int i = 0; i < num_contours; i++) {
    float ssid = 0.0F;
    cv::RotatedRect box = cv::minAreaRect(contours[i]);
    auto array = GetMiniBoxes(box, ssid);
    const auto box_for_unclip = array;

    if (ssid < min_size) {continue;}

    float score = 0.0F;
    if (use_polygon_score) {
      score = PolygonScoreAcc(contours[i], pred);
    } else {
      score = BoxScoreFast(array, pred);
    }
    if (score < box_thresh) {continue;}

    const cv::RotatedRect unclipped = UnClip(box_for_unclip, det_db_unclip_ratio);
    if (unclipped.size.height < 1.001F && unclipped.size.width < 1.001F) {
      continue;
    }

    auto cliparray = GetMiniBoxes(unclipped, ssid);
    if (ssid < min_size + 2) {continue;}

    std::vector<std::vector<int>> intcliparray;
    for (int num_pt = 0; num_pt < 4; num_pt++) {
      std::vector<int> a{
        static_cast<int>(clamp(
          std::round(cliparray[num_pt][0] / static_cast<float>(width) *
            static_cast<float>(dest_width)),
          0.0F, static_cast<float>(dest_width))),
        static_cast<int>(clamp(
          std::round(cliparray[num_pt][1] / static_cast<float>(height) *
            static_cast<float>(dest_height)),
          0.0F, static_cast<float>(dest_height)))};
      intcliparray.push_back(a);
    }
    boxes.push_back(intcliparray);
  }
  return boxes;
}

std::vector<std::vector<int>> PostProcessor::OrderPointsClockwise(
  std::vector<std::vector<int>> pts)
{
  auto box = pts;
  std::sort(box.begin(), box.end(), XsortInt);

  std::vector<std::vector<int>> leftmost = {box[0], box[1]};
  std::vector<std::vector<int>> rightmost = {box[2], box[3]};

  if (leftmost[0][1] > leftmost[1][1]) {std::swap(leftmost[0], leftmost[1]);}
  if (rightmost[0][1] > rightmost[1][1]) {std::swap(rightmost[0], rightmost[1]);}

  return {leftmost[0], rightmost[0], rightmost[1], leftmost[1]};
}

OCRBoxVec PostProcessor::FilterTagDetRes(
  std::vector<std::vector<std::vector<int>>> boxes,
  const cv::Mat & input_img)
{
  const int oriimg_h = input_img.rows;
  const int oriimg_w = input_img.cols;
  OCRBoxVec ocrboxes;
  for (auto & box : boxes) {
    box = OrderPointsClockwise(box);
    for (auto & pt : box) {
      pt[0] = imin(imax(pt[0], 0), oriimg_w - 1);
      pt[1] = imin(imax(pt[1], 0), oriimg_h - 1);
    }
  }

  for (const auto & box : boxes) {
    const int rect_width = static_cast<int>(std::sqrt(
      std::pow(box[0][0] - box[1][0], 2) + std::pow(box[0][1] - box[1][1], 2)));
    const int rect_height = static_cast<int>(std::sqrt(
      std::pow(box[0][0] - box[3][0], 2) + std::pow(box[0][1] - box[3][1], 2)));
    if (rect_width <= 3 || rect_height <= 3) {continue;}
    OCRBox b;
    b.x1 = box[0][0];
    b.y1 = box[0][1];
    b.x2 = box[1][0];
    b.y2 = box[1][1];
    b.x3 = box[2][0];
    b.y3 = box[2][1];
    b.x4 = box[3][0];
    b.y4 = box[3][1];
    ocrboxes.push_back(b);
  }
  return ocrboxes;
}

}  // namespace topsbot_ppocr