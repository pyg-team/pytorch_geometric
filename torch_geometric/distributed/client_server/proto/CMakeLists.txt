find_package(protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)

set(PROTO_FILES proto/common.proto proto/service.proto)

set(_PROTOBUF_PROTOC $<TARGET_FILE:protobuf::protoc>)
set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:gRPC::grpc_cpp_plugin>)

# Generated directories:
set(CPP_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/Generated")
set(PY_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/Python")
file(MAKE_DIRECTORY ${CPP_GENERATED_DIR})
file(MAKE_DIRECTORY ${PY_GENERATED_DIR})

# Python package is called 'client_server_proto'
file(MAKE_DIRECTORY "${PY_GENERATED_DIR}/client_server_proto")

# TODO(manan): this doesn't work due to relative path issues when generating C++
# and Python code; as a result, we default to the below implementation.
#
# get_target_property(grpc_cpp_plugin_location gRPC::grpc_cpp_plugin LOCATION)
# protobuf_generate(TARGET proto LANGUAGE cpp) protobuf_generate(TARGET proto
# LANGUAGE grpc GENERATE_EXTENSIONS .grpc.pb.h .grpc.pb.cc PLUGIN
# "protoc-gen-grpc=${grpc_cpp_plugin_location}")
foreach(proto ${PROTO_FILES})
  # Name: e.g. "common" for file "common.proto"
  get_filename_component(proto_name "${proto}" NAME_WE)
  # Path: absolute path to the proto file
  get_filename_component(proto_path "${proto}" ABSOLUTE)
  # Directory containing the proto files:
  get_filename_component(proto_dir "${proto_path}" PATH)

  # Generate C++ headers and sources; these will be depended on by
  # other targets in client and server:
  set(proto_srcs "${CPP_GENERATED_DIR}/${proto_name}.pb.cc")
  set(proto_hdrs "${CPP_GENERATED_DIR}/${proto_name}.pb.h")
  set(grpc_srcs "${CPP_GENERATED_DIR}/${proto_name}.grpc.pb.cc")
  set(grpc_hdrs "${CPP_GENERATED_DIR}/${proto_name}.grpc.pb.h")

  add_custom_command(
    OUTPUT "${proto_srcs}" "${proto_hdrs}" "${grpc_srcs}" "${grpc_hdrs}"
    COMMAND
      ${_PROTOBUF_PROTOC} ARGS --grpc_out="${CPP_GENERATED_DIR}"
      --cpp_out="${CPP_GENERATED_DIR}" -I="${proto_dir}"
      --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
      "${proto_name}.proto"
    DEPENDS "${proto_path}")

  # Generate Python libraries; these will be installed as a standalone
  # package named client_server_proto that will be imported by Python
  # tools:
  add_custom_command(
    OUTPUT "${PY_GENERATED_DIR}/client_server_proto/${proto_name}_pb2_grpc.py"
           "${PY_GENERATED_DIR}/client_server_proto/${proto_name}_pb2.py"
    COMMAND
      python3 ARGS -m grpc_tools.protoc -I "${proto_dir}" --python_out
      "${PY_GENERATED_DIR}/client_server_proto" --grpc_python_out
      "${PY_GENERATED_DIR}/client_server_proto" --pyi_out "${PY_GENERATED_DIR}/client_server_proto"
      "${proto_path}"
    DEPENDS "${proto_path}")
  list(APPEND GRPC_PYTHON_GENERATED_SOURCES
       "${PY_GENERATED_DIR}/client_server_proto/${proto_name}_pb2_grpc.py")
  list(APPEND GRPC_HEADER_FILES ${proto_hdrs} ${grpc_hdrs})
  list(APPEND GRPC_CPP_SOURCES ${proto_srcs} ${grpc_srcs})
endforeach(proto)

# Python generation target:
set_source_files_properties(${GRPC_PYTHON_GENERATED_SOURCES}
                            PROPERTIES GENERATED TRUE)
add_custom_target(proto_py ALL SOURCES ${GRPC_PYTHON_GENERATED_SOURCES})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.cmake
               ${PY_GENERATED_DIR}/setup.py)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/__init__.py.cmake
               "${PY_GENERATED_DIR}/client_server_proto/__init__.py")

find_package(Python3 REQUIRED)
add_custom_target(proto_py_install ALL)
add_custom_command(TARGET proto_py_install COMMAND python3 ARGS -m pip install
                                             grpcio-tools)
add_custom_command(TARGET proto_py_install COMMAND python3 ARGS -m pip install
  ${PY_GENERATED_DIR})

# C++ generation target:
add_library(proto OBJECT ${GRPC_HEADER_FILES} ${GRPC_CPP_SOURCES})
target_link_libraries(proto PUBLIC protobuf::libprotobuf gRPC::grpc
                                   gRPC::grpc++)
target_include_directories(proto PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
