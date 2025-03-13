#include "encoder.h"
#include "Generated/common.pb.h"
#include "arrow/array.h"
#include "arrow/type_fwd.h"
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>

namespace server {

std::string ByteEncoder::encode(int v) {
  std::string ret(INT_BYTE_LEN, 0x00);
  char *pos = const_cast<char *>(ret.c_str());

  // Encode integer in a way that key is fixed length bytes and ordered.
  size_t bytes_to_fill = std::min(INT_BYTE_LEN, 8);
  for (size_t i = 1; i <= bytes_to_fill; ++i) {
    pos[i - 1] = (v >> ((bytes_to_fill - i) << 3)) & 0xFF;
  }

  return ret;
}

void ByteEncoder::encode(std::shared_ptr<arrow::Array> &a,
                         std::vector<featurestore::FeatureList> *out,
                         size_t start, size_t size) {
  arrow::Type::type t = a->type_id();
  if (t == arrow::Type::INT64) {
    auto t_a = std::static_pointer_cast<arrow::Int64Array>(a);
    ByteEncoder::encode_internal(t_a, a->length(), out, start);
  } else if (t == arrow::Type::INT32) {
    auto t_a = std::static_pointer_cast<arrow::Int32Array>(a);
    ByteEncoder::encode_internal(t_a, a->length(), out, start);
  } else if (t == arrow::Type::BOOL) {
    auto t_a = std::static_pointer_cast<arrow::BooleanArray>(a);
    ByteEncoder::encode_internal(t_a, a->length(), out, start);
  } else if (t == arrow::Type::FLOAT) {
    auto t_a = std::static_pointer_cast<arrow::FloatArray>(a);
    ByteEncoder::encode_internal(t_a, a->length(), out, start);
  } else if (t == arrow::Type::DOUBLE) {
    auto t_a = std::static_pointer_cast<arrow::DoubleArray>(a);
    ByteEncoder::encode_internal(t_a, a->length(), out, start);
  } else if (t == arrow::Type::TIMESTAMP) {
    auto t_a = std::static_pointer_cast<arrow::TimestampArray>(a);
    ByteEncoder::encode_internal(t_a, a->length(), out, start);
  } else if (t == arrow::Type::DATE32) {
    auto t_a = std::static_pointer_cast<arrow::Date32Array>(a);
    ByteEncoder::encode_internal(t_a, a->length(), out, start);
  } else if (t == arrow::Type::DATE64) {
    auto t_a = std::static_pointer_cast<arrow::Date64Array>(a);
    ByteEncoder::encode_internal(t_a, a->length(), out, start);
  } else if (t == arrow::Type::TIME64) {
    auto t_a = std::static_pointer_cast<arrow::Time64Array>(a);
    ByteEncoder::encode_internal(t_a, a->length(), out, start);
  } else if (t == arrow::Type::STRING) {
    auto t_a = std::static_pointer_cast<arrow::StringArray>(a);
    ByteEncoder::encode_internal(t_a, a->length(), out, start);
  }
  else {
    throw new std::runtime_error("Unsupported type");
  }
}

template <typename T>
void ByteEncoder::encode_internal(std::shared_ptr<T> &t, size_t len,
                                  std::vector<featurestore::Tensor> *out,
                                  size_t start) {
  for (size_t i = start; i < start + len; i++) {
    featurestore::Feature *f = (*out)[i - start].add_feature();
    float val;
    if constexpr (IsIntArray<T>()) {
      val = static_cast<float>(t->Value(i));
    } else if constexpr (IsFloatArray<T>()) {
      val = static_cast<float>(t->Value(i));
    }
    f->mutable_value_list()->add_value(val);
  }
}

} // namespace server
