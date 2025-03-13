#pragma once

#include "Generated/common.pb.h"
#include "arrow/type_fwd.h"
#include <string>
#include <vector>

namespace server {

const int INT_BYTE_LEN = 8;

class ByteEncoder {
public:
  /* Primary keys can either be integers or strings. If
   * they are strings, they are left as-is. If they are
   * integers, they are encoded.*/
  std::string encode(int);

  /* Non-primary key columns are encoded via the protobuf
   * Tensor representation.*/
  void encode(std::shared_ptr<arrow::Array> &,
              std::vector<featurestore::Tensor> *, size_t, size_t);

private:
  template <typename T>
  void encode_internal(std::shared_ptr<T> &, size_t,
                       std::vector<featurestore::Tensor> *, size_t);
};

template <typename T> constexpr bool IsIntArray() {
  return (std::is_same_v<T, arrow::Int64Array> ||
          std::is_same_v<T, arrow::Int32Array> ||
          std::is_same_v<T, arrow::BooleanArray> ||
          std::is_same_v<T, arrow::TimestampArray> ||
          std::is_same_v<T, arrow::Date32Array> ||
          std::is_same_v<T, arrow::Date64Array> ||
          std::is_same_v<T, arrow::Time64Array>);
}

template <typename T> constexpr bool IsFloatArray() {
  return (std::is_same_v<T, arrow::FloatArray> ||
          std::is_same_v<T, arrow::DoubleArray>);
}

} // namespace server
