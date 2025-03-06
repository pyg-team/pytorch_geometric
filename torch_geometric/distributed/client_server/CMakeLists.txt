cmake_minimum_required(VERSION 3.13)
project(client_server LANGUAGES CXX)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
set(CMAKE_CXX_STANDARD 17)

# NOTE: remove when building for production
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Debug)

add_subdirectory(proto)
add_subdirectory(client)
add_subdirectory(server)
