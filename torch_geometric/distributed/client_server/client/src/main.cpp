#include <Generated/service.grpc.pb.h>
#include <Generated/service.pb.h>
#include <grpc/grpc.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>

#include <iostream>

// TODO(manan): these should be arguments, and more general:
const std::string CUSTOMER_PATH = "";
const std::string STOCK_PATH = "";

int main(int argc, char *argv[]) {
  // Call
  auto channel = grpc::CreateChannel("localhost:50051",
                                     grpc::InsecureChannelCredentials());
  std::unique_ptr<featurestore::FeatureStore::Stub> stub =
      featurestore::FeatureStore::NewStub(channel);

  // Load 2 node types:
  {
    featurestore::LoadFeaturesRequest load_req;
    featurestore::EmptyResponse load_res;
    load_req.set_type("customer");
    load_req.set_path(CUSTOMER_PATH);
    grpc::ClientContext ctx;
    grpc::Status status = stub->LoadFeatures(&ctx, load_req, &load_res);
  }
  {
    featurestore::LoadFeaturesRequest load_req;
    featurestore::EmptyResponse load_res;
    load_req.set_type("stock");
    load_req.set_path(STOCK_PATH);
    grpc::ClientContext ctx;
    grpc::Status status = stub->LoadFeatures(&ctx, load_req, &load_res);
  }

  // Fetch:
  {
    featurestore::KeysRequest fetch_req;
    featurestore::KeysResponse fetch_res;
    fetch_req.set_type("customer");
    fetch_req.add_idx(1);
    fetch_req.add_idx(10);
    grpc::ClientContext ctx;
    grpc::Status status = stub->GetFeatures(&ctx, fetch_req, &fetch_res);

    // Output result
    std::cout << status.ok() << std::endl;
    std::cout << "Response: " << fetch_res.DebugString() << std::endl;
  }
  {
    featurestore::KeysRequest fetch_req;
    featurestore::KeysResponse fetch_res;
    fetch_req.set_type("stock");
    fetch_req.add_idx(1);
    fetch_req.add_idx(10);
    grpc::ClientContext ctx;
    grpc::Status status = stub->GetFeatures(&ctx, fetch_req, &fetch_res);

    // Output result
    std::cout << status.ok() << std::endl;
    std::cout << "Response: " << fetch_res.DebugString() << std::endl;
  }

  return 0;
}
