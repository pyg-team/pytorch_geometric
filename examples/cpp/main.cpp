#include <torch/script.h>
#include <torchscatter/scatter.h>
#include <torchsparse/sparse.h>

#include <iostream>

int main(int argc, const char *argv[]) {
  if (argc != 2) {
    std::cerr << "usage: hello-world <path-to-exported-script-module>\n";
    return -1;
  }

  torch::jit::script::Module model;
  try {
    model = torch::jit::load(argv[1]);
  } catch (const c10::Error &e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  auto x = torch::randn({5, 32});
  auto edge_index = torch::tensor({
      {0, 1, 1, 2, 2, 3, 3, 4},
      {1, 0, 2, 1, 3, 2, 4, 3},
  });

  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(x);
  inputs.push_back(edge_index);

  auto out = model.forward(inputs).toTensor();
  std::cout << "output tensor shape: " << out.sizes() << std::endl;
}
