// Copyright 2026 TOPSBOT
// SPDX-License-Identifier: Apache-2.0

#include "topsbot_ppocr/ppocr_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>

namespace topsbot_ppocr
{

size_t tensor_element_count(const taconn_inout_attr_t & attr)
{
  size_t count = 1U;
  for (unsigned int i = 0; i < attr.dim_count; ++i) {
    count *= attr.dim_size[i];
  }
  return count;
}

size_t tensor_byte_size(const taconn_inout_attr_t & attr)
{
  const size_t count = tensor_element_count(attr);
  switch (attr.data_format) {
    case TACONN_DATA_FORMAT_FP32:
    case TACONN_DATA_FORMAT_INT32:
    case TACONN_DATA_FORMAT_UINT32:
      return sizeof(uint32_t) * count;
    case TACONN_DATA_FORMAT_FP16:
    case TACONN_DATA_FORMAT_BFP16:
    case TACONN_DATA_FORMAT_INT16:
    case TACONN_DATA_FORMAT_UINT16:
      return sizeof(uint16_t) * count;
    case TACONN_DATA_FORMAT_UINT8:
    case TACONN_DATA_FORMAT_INT8:
    case TACONN_DATA_FORMAT_CHAR:
    case TACONN_DATA_FORMAT_BOOL8:
      return sizeof(uint8_t) * count;
    case TACONN_DATA_FORMAT_FP64:
    case TACONN_DATA_FORMAT_INT64:
    case TACONN_DATA_FORMAT_UINT64:
      return sizeof(uint64_t) * count;
    case TACONN_DATA_FORMAT_INT4:
    case TACONN_DATA_FORMAT_UINT4:
      return (count + 1U) / 2U;
    default:
      return 0U;
  }
}

uint16_t float_to_half(float value)
{
  const uint32_t bits = [&]() {
    uint32_t b;
    std::memcpy(&b, &value, sizeof(b));
    return b;
  }();
  const uint16_t sign = static_cast<uint16_t>((bits >> 31) & 0x1U);
  int exponent = static_cast<int>((bits >> 23) & 0xFFU);
  uint32_t fraction = bits & 0x7FFFFFU;

  if (exponent == 0 && fraction == 0) {
    return static_cast<uint16_t>(sign << 15);
  }
  if (exponent == 0xFF) {
    if (fraction == 0) {
      return static_cast<uint16_t>((sign << 15) | 0x7C00);
    }
    return static_cast<uint16_t>((sign << 15) | 0x7E00);
  }

  exponent -= 127;
  if (exponent < -14) {
    fraction = (0x800000U + fraction) >> (13 - exponent - 14);
    fraction |= (fraction >> 13) & 1U;
    return static_cast<uint16_t>((sign << 15) | fraction);
  }
  if (exponent > 15) {
    return static_cast<uint16_t>((sign << 15) | 0x7C00);
  }

  exponent += 15;
  fraction >>= 13;
  if (fraction & 0x1000U) {
    fraction += 1U;
    if (fraction & 0x8000U) {
      fraction >>= 1;
      exponent += 1;
    }
  }
  return static_cast<uint16_t>((sign << 15) | (exponent << 10) | (fraction & 0x3FFU));
}

float half_to_float(uint16_t value)
{
  const uint32_t sign = (value >> 15) & 0x1U;
  uint32_t exponent = (value >> 10) & 0x1FU;
  const uint32_t mantissa = value & 0x3FFU;
  uint32_t bits = 0U;

  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign << 31;
      float zero;
      std::memcpy(&zero, &bits, sizeof(zero));
      return zero;
    }
    // Subnormal half: normalize then reconstruct via the normal path.
    uint32_t m = mantissa;
    uint32_t e = 127U - 15U + 1U;  // smallest normal exponent in float bias
    while ((m & 0x400U) == 0U) {
      m <<= 1;
      --e;
    }
    m &= 0x3FFU;
    bits = (sign << 31) | (e << 23) | (m << 13);
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
  }
  if (exponent == 31) {
    bits = (sign << 31) | 0x7F800000U | (mantissa << 13);
  } else {
    bits = (sign << 31) | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
  }
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

int32_t fp32_to_dfp(float value, const int8_t fixed_point_pos, taconn_data_format_t type)
{
  int32_t min_range = 0;
  int32_t max_range = 0;
  switch (type) {
    case TACONN_DATA_FORMAT_INT8:
      min_range = -128; max_range = 127; break;
    case TACONN_DATA_FORMAT_INT16:
      min_range = -32768; max_range = 32767; break;
    case TACONN_DATA_FORMAT_INT32:
      min_range = std::numeric_limits<int32_t>::min();
      max_range = std::numeric_limits<int32_t>::max();
      break;
    case TACONN_DATA_FORMAT_UINT8:
      min_range = 0; max_range = 255; break;
    case TACONN_DATA_FORMAT_UINT16:
      min_range = 0; max_range = 65535; break;
    case TACONN_DATA_FORMAT_UINT32:
      min_range = 0; max_range = std::numeric_limits<int32_t>::max(); break;
    case TACONN_DATA_FORMAT_INT4:
      min_range = -8; max_range = 7; break;
    case TACONN_DATA_FORMAT_UINT4:
      min_range = 0; max_range = 15; break;
    case TACONN_DATA_FORMAT_CHAR:
      min_range = -128; max_range = 127; break;
    case TACONN_DATA_FORMAT_BOOL8:
      min_range = 0; max_range = 1; break;
    default:
      min_range = std::numeric_limits<int32_t>::min();
      max_range = std::numeric_limits<int32_t>::max();
      break;
  }

  int32_t data = 0;
  if (fixed_point_pos > 0) {
    data = static_cast<int32_t>(std::rint(value * static_cast<float>(1 << fixed_point_pos)));
  } else if (fixed_point_pos < 0) {
    data = static_cast<int32_t>(
      std::rint(value / static_cast<float>(1 << (-fixed_point_pos))));
  } else {
    data = static_cast<int32_t>(std::rint(value));
  }
  return std::clamp(data, min_range, max_range);
}

float dfp_to_fp32(const int32_t value, const int8_t fixed_point_pos)
{
  if (fixed_point_pos > 0) {
    return static_cast<float>(value) /
           static_cast<float>(static_cast<int64_t>(1) << fixed_point_pos);
  }
  return static_cast<float>(value) *
         static_cast<float>(static_cast<int64_t>(1) << (-fixed_point_pos));
}

void quantize_input(
  const float * src, void * dst, const size_t num_elements,
  const taconn_inout_attr_t & attr)
{
  if (attr.quant_format == TACONN_QNT_TYPE_NONE) {
    if (attr.data_format == TACONN_DATA_FORMAT_FP32) {
      std::memcpy(dst, src, num_elements * sizeof(float));
      return;
    }
    if (attr.data_format == TACONN_DATA_FORMAT_FP16) {
      auto * dst_fp16 = static_cast<uint16_t *>(dst);
      for (size_t i = 0; i < num_elements; i++) {
        dst_fp16[i] = float_to_half(src[i]);
      }
      return;
    }
    return;
  }

  if (attr.quant_format == TACONN_QNT_TYPE_ASYMMETRIC) {
    const float scale = attr.quant_data.affine.tf_scale;
    const int32_t zero_point = attr.quant_data.affine.tf_zero_point;
    if (attr.data_format == TACONN_DATA_FORMAT_INT8) {
      auto * dst_i8 = static_cast<int8_t *>(dst);
      for (size_t i = 0; i < num_elements; i++) {
        const int32_t q = static_cast<int32_t>(std::lrint(src[i] / scale)) + zero_point;
        dst_i8[i] = static_cast<int8_t>(std::clamp(q, -128, 127));
      }
    } else if (attr.data_format == TACONN_DATA_FORMAT_UINT8) {
      auto * dst_u8 = static_cast<uint8_t *>(dst);
      for (size_t i = 0; i < num_elements; i++) {
        const int32_t q = static_cast<int32_t>(std::lrint(src[i] / scale)) + zero_point;
        dst_u8[i] = static_cast<uint8_t>(std::clamp(q, 0, 255));
      }
    } else if (attr.data_format == TACONN_DATA_FORMAT_FP16) {
      auto * dst_fp16 = static_cast<uint16_t *>(dst);
      for (size_t i = 0; i < num_elements; i++) {
        dst_fp16[i] = float_to_half(src[i]);
      }
    }
    return;
  }

  if (attr.quant_format == TACONN_QNT_TYPE_DFP) {
    const int8_t fixed_point_pos = attr.quant_data.dfp.fixed_point_pos;
    if (attr.data_format == TACONN_DATA_FORMAT_INT8) {
      auto * dst_i8 = static_cast<int8_t *>(dst);
      for (size_t i = 0; i < num_elements; i++) {
        dst_i8[i] = static_cast<int8_t>(
          fp32_to_dfp(src[i], fixed_point_pos, static_cast<taconn_data_format_t>(attr.data_format)));
      }
    } else if (attr.data_format == TACONN_DATA_FORMAT_INT16) {
      auto * dst_i16 = static_cast<int16_t *>(dst);
      for (size_t i = 0; i < num_elements; i++) {
        dst_i16[i] = static_cast<int16_t>(
          fp32_to_dfp(src[i], fixed_point_pos, static_cast<taconn_data_format_t>(attr.data_format)));
      }
    }
    return;
  }
}

void dequantize_output(
  const void * src, const taconn_inout_attr_t & attr, std::vector<float> & dst)
{
  const size_t num = tensor_element_count(attr);
  dst.resize(num);

  const int32_t zp = attr.quant_format == TACONN_QNT_TYPE_ASYMMETRIC
    ? attr.quant_data.affine.tf_zero_point : 0;
  const float scale = attr.quant_format == TACONN_QNT_TYPE_ASYMMETRIC
    ? attr.quant_data.affine.tf_scale : 1.0F;

  switch (attr.data_format) {
    case TACONN_DATA_FORMAT_FP32: {
      std::memcpy(dst.data(), src, num * sizeof(float));
      break;
    }
    case TACONN_DATA_FORMAT_FP16: {
      const auto * s = static_cast<const uint16_t *>(src);
      for (size_t i = 0; i < num; i++) {dst[i] = half_to_float(s[i]);}
      break;
    }
    case TACONN_DATA_FORMAT_INT8: {
      const auto * s = static_cast<const int8_t *>(src);
      for (size_t i = 0; i < num; i++) {
        dst[i] = (static_cast<float>(s[i]) - static_cast<float>(zp)) * scale;
      }
      break;
    }
    case TACONN_DATA_FORMAT_UINT8: {
      const auto * s = static_cast<const uint8_t *>(src);
      for (size_t i = 0; i < num; i++) {
        dst[i] = (static_cast<float>(s[i]) - static_cast<float>(zp)) * scale;
      }
      break;
    }
    case TACONN_DATA_FORMAT_INT16: {
      const auto * s = static_cast<const int16_t *>(src);
      if (attr.quant_format == TACONN_QNT_TYPE_DFP) {
        const int8_t fl = attr.quant_data.dfp.fixed_point_pos;
        for (size_t i = 0; i < num; i++) {dst[i] = dfp_to_fp32(s[i], fl);}
      } else {
        for (size_t i = 0; i < num; i++) {
          dst[i] = (static_cast<float>(s[i]) - static_cast<float>(zp)) * scale;
        }
      }
      break;
    }
    default:
      dst.assign(num, 0.0F);
      break;
  }
}

bool read_dict(const std::string & path, std::vector<std::string> & chars, std::string & error)
{
  std::ifstream in(path);
  if (!in) {
    error = "failed to open recognition dictionary: " + path;
    return false;
  }
  chars.clear();
  chars.emplace_back("#");  // CTC blank
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') {line.pop_back();}
    chars.push_back(line);
  }
  chars.emplace_back(" ");  // space
  return true;
}

}  // namespace topsbot_ppocr