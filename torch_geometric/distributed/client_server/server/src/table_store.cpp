#include "table_store.h"
#include "Generated/common.pb.h"
#include "arrow/chunked_array.h"
#include "arrow/compute/expression.h"
#include "arrow/dataset/dataset.h"
#include "arrow/dataset/discovery.h"
#include "arrow/dataset/file_parquet.h"
#include "arrow/dataset/scanner.h"
#include "arrow/filesystem/filesystem.h"
#include "arrow/filesystem/localfs.h"
#include "arrow/io/api.h"
#include "arrow/table.h"
#include "arrow/type.h"
#include "rocksdb/db.h"
#include "rocksdb/options.h"
#include <exception>
#include <filesystem>
#include <iostream>

namespace server {

std::string generate_random_string(size_t length) {
  static const std::string characters =
      "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
  std::random_device random_device;
  std::mt19937 generator(random_device());
  std::uniform_int_distribution<> distribution(0, characters.size() - 1);

  std::string random_string;
  for (size_t i = 0; i < length; ++i) {
    random_string += characters[distribution(generator)];
  }
  return random_string;
}

RocksDBTableStore::RocksDBTableStore(std::string db_path) {
  rocksdb::Options options;
  options.IncreaseParallelism();
  options.OptimizeLevelStyleCompaction();
  options.create_if_missing = true;
  rocksdb::Status s = rocksdb::DB::Open(options, db_path, &m_db);
  m_dir = db_path;
  assert(s.ok());
};

std::string RocksDBTableStore::Get(const std::string key) {
  std::string value;
  rocksdb::Status s = m_db->Get(rocksdb::ReadOptions(), key, &value);
  assert(s.ok());
  return value;
}

std::vector<std::string>
RocksDBTableStore::MultiGet(const std::vector<std::string> keys) {
  std::vector<std::string> values(keys.size());
  std::vector<rocksdb::Slice> k_slices;
  for (const auto &k : keys) {
    k_slices.emplace_back(k.data(), k.size());
  }
  std::vector<rocksdb::Status> out =
      m_db->MultiGet(rocksdb::ReadOptions(), k_slices, &values);
  for (const auto &s : out) {
    assert(s.ok());
  }
  return values;
}

void RocksDBTableStore::BulkLoadParquet(const std::string dir) {
  // Read parquet dataset into arrow Table:
  std::shared_ptr<arrow::fs::FileSystem> filesystem =
      std::make_shared<arrow::fs::LocalFileSystem>();
  std::shared_ptr<arrow::dataset::ParquetFileFormat> format =
      std::make_shared<arrow::dataset::ParquetFileFormat>();

  arrow::fs::FileSelector selector;
  selector.base_dir = dir;
  selector.recursive = true;

  arrow::dataset::FileSystemFactoryOptions options;
  std::shared_ptr<arrow::dataset::DatasetFactory> dataset_factory(
      arrow::dataset::FileSystemDatasetFactory::Make(filesystem, selector,
                                                     format, options)
          .ValueOrDie());

  std::shared_ptr<arrow::dataset::Dataset> dataset(
      dataset_factory->Finish().ValueOrDie());

  arrow::dataset::ScannerBuilder scanner_builder(dataset);
  arrow::Status s = scanner_builder.UseThreads(true);
  assert(s.ok());
  std::shared_ptr<arrow::dataset::Scanner> scanner(
      scanner_builder.Finish().ValueOrDie());
  std::shared_ptr<arrow::Table> table(scanner->ToTable().ValueOrDie());

  // Serialize arrow Table into Feature lists that we persist in
  // RocksDB:
  std::vector<featurestore::FeatureList> out(table->num_rows());
  SerializeArrowTable(table, &out);

  // Write feature lists to SST (write elsewhere, then ingest external file):
  rocksdb::Options out_options;
  rocksdb::SstFileWriter writer(rocksdb::EnvOptions(), out_options,
                                out_options.comparator);
  std::filesystem::path base(m_dir);
  base = base / (generate_random_string(8) + ".sst");
  rocksdb::Status status = writer.Open(base.string());
  assert(status.ok());

  for (size_t i = 0; i < table->num_rows(); ++i) {
    auto value = out[i].SerializeAsString();
    status = writer.Put(m_enc.encode(i), value);
    assert(status.ok());
  }
  status = writer.Finish();
  assert(status.ok());

  // Ingest external files:
  rocksdb::IngestExternalFileOptions ingest_options;
  status = m_db->IngestExternalFile({base.string()}, ingest_options);
  assert(status.ok());

  // Perform compaction per table:
  status = m_db->CompactRange(rocksdb::CompactRangeOptions(), nullptr, nullptr);
  assert(status.ok());
  std::filesystem::remove_all(base);
}

void RocksDBTableStore::SerializeArrowTable(
    std::shared_ptr<arrow::Table> t,
    std::vector<featurestore::FeatureList> *out) {
  std::shared_ptr<arrow::Schema> s(t->schema());

  for (const auto &field : s->field_names()) {
    std::shared_ptr<arrow::ChunkedArray> col(t->GetColumnByName(field));
    size_t start_idx = 0;

    // Serialize chunks in sequence. Convert column store to row store:
    for (const std::shared_ptr<arrow::Array> &chunk : col->chunks()) {
      RocksDBTableStore::SerializeArrowArray(chunk, out, start_idx);
      start_idx += chunk->length();
    }
  }
}

void RocksDBTableStore::SerializeArrowArray(
    std::shared_ptr<arrow::Array> chunk,
    std::vector<featurestore::FeatureList> *out, size_t start) {
  m_enc.encode(chunk, out, start, (size_t)chunk->length());
}

RocksDBTableStore::~RocksDBTableStore() {
  // TODO(manan): check status
  m_db->Close();
}

} // namespace server
