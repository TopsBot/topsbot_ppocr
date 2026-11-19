// Copyright 2026 TOPSBOT
// SPDX-License-Identifier: Apache-2.0

#ifndef TOPSBOT_PPOCR__PPOCR_UTILS_HPP_
#define TOPSBOT_PPOCR__PPOCR_UTILS_HPP_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <ta-runtime-api.h>

namespace topsbot_ppocr
{

/// Product of all dimensions in an in/out attr.
size_t tensor_element_count(const taconn_inout_attr_t & attr);

/// Byte size of a tensor described by attr, given its data_format.
size_t tensor_byte_size(const taconn_inout_attr_t & attr);

/// IEEE 754 half <-> single conversions (correct rounding for normals).
uint16_t float_to_half(float value);
float half_to_float(uint16_t value);

/// Fixed-point (DFP) quantize / dequantize helpers.
int32_t fp32_to_dfp(float value, int8_t fixed_point_pos, taconn_data_format_t type);
float dfp_to_fp32(int32_t value, int8_t fixed_point_pos);

/// Quantize a normalized float tensor (already mean/std scaled, CHW) into the
/// NPU input buffer according to the input tensor's data/quant format.
void quantize_input(
  const float * src, void * dst, size_t num_elements, const taconn_inout_attr_t & attr);

/// Dequantize an NPU output buffer into a float vector.
void dequantize_output(
  const void * src, const taconn_inout_attr_t & attr, std::vector<float> & dst);

/// Load the recognition dictionary. Mirrors the model-zoo convention of
/// prepending a CTC blank ("#") and appending a space character.
bool read_dict(const std::string & path, std::vector<std::string> & chars, std::string & error);

}  // namespace topsbot_ppocr

#endif  // TOPSBOT_PPOCR__PPOCR_UTILS_HPP_