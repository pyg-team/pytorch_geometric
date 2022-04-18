cmake_minimum_required(VERSION 3.10)
project(hello-world)

# The first thing do is to tell cmake to find the TorchScatter
# and TorchSparse libraries. The package pulls in all the necessary
# torch libraries, so there is no need to add `find_package(Torch)`.
find_package(TorchScatter REQUIRED)
find_package(TorchSparse REQUIRED)

find_package(Python3 COMPONENTS Development)

add_executable(hello-world main.cpp)

# We now need to link the TorchScatter and TorchSparse libraries
# to our executable. We can do that by using the
# TorchScatter::TorchScatter and TorchSparse::TorchSparse targets,
# which also adds all the necessary torch dependencies.
target_compile_features(hello-world PUBLIC cxx_range_for)
target_link_libraries(hello-world TorchScatter::TorchScatter)
target_link_libraries(hello-world TorchSparse::TorchSparse)
set_property(TARGET hello-world PROPERTY CXX_STANDARD 14)
