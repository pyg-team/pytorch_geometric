#include "encoder.h"
#include "rocksdb/db.h"
#include <arrow/type_fwd.h>
#include <memory>
#include <string>
#include <vector>

namespace server {
class TableStore {
public:
  virtual std::string Get(const std::string key) = 0;
  virtual std::vector<std::string>
  MultiGet(const std::vector<std::string> keys) = 0;
  virtual void BulkLoadParquet(const std::string path) = 0;
  virtual ~TableStore() = default;
};

class RocksDBTableStore : public TableStore {
public:
  RocksDBTableStore(std::string path);
  std::string Get(const std::string key);
  std::vector<std::string> MultiGet(const std::vector<std::string> keys);
  void BulkLoadParquet(const std::string path);
  ~RocksDBTableStore();

private:
  std::string m_dir;
  rocksdb::DB *m_db;
  ByteEncoder m_enc;
  void SerializeArrowTable(std::shared_ptr<arrow::Table>,
                           std::vector<featurestore::FeatureList> *);
  void SerializeArrowArray(std::shared_ptr<arrow::Array>,
                           std::vector<featurestore::FeatureList> *, size_t);
};
} // namespace server
