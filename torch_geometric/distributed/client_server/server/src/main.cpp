#include <Generated/service.grpc.pb.h>
#include <Generated/service.pb.h>

#include <grpc/grpc.h>
#include <grpcpp/server_builder.h>

#include "rocksdb/db.h"
#include "rocksdb/options.h"
#include "rocksdb/slice.h"

#include <filesystem>
#include <iostream>
#include <stdexcept>

#include "encoder.h"
#include "table_store.h"

class FeatureStoreService final : public featurestore::FeatureStore::Service {
public:
  FeatureStoreService(std::string base)
      : featurestore::FeatureStore::Service() {
    m_base = base;
    std::filesystem::create_directory(base);
  }

  virtual grpc::Status
  LoadFeatures(grpc::ServerContext *context,
               const featurestore::LoadFeaturesRequest *request,
               featurestore::EmptyResponse *response) {
    if (m_db.find(request->type()) != m_db.end()) {
      throw new std::runtime_error("Cannot insert same node type twice.");
    }
    std::filesystem::path db_path(m_base);
    db_path = db_path / request->type();

    // TODO(manan): fix?
    std::filesystem::remove_all(db_path);
    std::cout << "Initializing RocksDB database at " << db_path << "...";

    server::TableStore *t = new server::RocksDBTableStore(db_path.string());
    t->BulkLoadParquet(request->path());
    m_db[request->type()] = t;
    std::cout << "done." << std::endl;
    return grpc::Status::OK;
  }

  virtual grpc::Status GetFeatures(grpc::ServerContext *context,
                                   const featurestore::KeysRequest *request,
                                   featurestore::KeysResponse *response) {
    std::cout << "Received request " << request->ShortDebugString()
              << std::endl;
    response->set_type(request->type());
    server::TableStore *t = m_db[request->type()];

    // TODO(manan): cleanup...
    std::vector<std::string> idxs;
    idxs.reserve(request->idx_size());
    std::transform(request->idx().begin(), request->idx().end(),
                   std::back_inserter(idxs),
                   [this](int32_t x) { return m_enc.encode(x); });
    std::vector<std::string> out = t->MultiGet(idxs);
    for (const auto &fl_enc : out) {
      featurestore::FeatureList fl;
      fl.ParseFromString(fl_enc);
      response->mutable_features()->Add(std::move(fl));
    }
    return grpc::Status::OK;
  }

private:
  std::unordered_map<std::string, server::TableStore *> m_db;
  std::string m_base;
  ByteEncoder m_enc;
};

int main(int argc, char *argv[]) {
  grpc::ServerBuilder builder;
  builder.AddListeningPort("0.0.0.0:50051", grpc::InsecureServerCredentials());

  FeatureStoreService my_service("/tmp/feature_store");
  builder.RegisterService(&my_service);

  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  server->Wait();

  return 0;
}
