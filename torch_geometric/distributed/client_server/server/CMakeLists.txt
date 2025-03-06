set(SOURCES
    src/main.cpp
)

include(ExternalProject)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
set(ARCH native CACHE STRING "CPU to build for: -march value or 'default' to not pass -march at all")

set(RocksDB_CMAKE_ARGS
  -DUSE_RTTI=1
  -DPORTABLE=${PORTABLE_ROCKSDB}
  -DFORCE_SSE42=${ROCKSDB_SSE42}
  -DFORCE_AVX=${ROCKSDB_AVX}
  -DFORCE_AVX2=${ROCKSDB_AVX2}
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
  -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
  -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}
  -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
  -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
  -DCMAKE_SHARED_LINKER_FLAGS=${CMAKE_SHARED_LINKER_FLAGS}
  -DCMAKE_STATIC_LINKER_FLAGS=${CMAKE_STATIC_LINKER_FLAGS}
  -DCMAKE_EXE_LINKER_FLAGS=${CMAKE_EXE_LINKER_FLAGS}
  -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  -DFAIL_ON_WARNINGS=OFF
  -DWITH_GFLAGS=OFF
  -DWITH_TESTS=OFF
  -DWITH_TOOLS=OFF
  -DWITH_CORE_TOOLS=OFF
  -DWITH_BENCHMARK_TOOLS=OFF
  -DWITH_BZ2=OFF
  -DWITH_LZ4=OFF
  -DWITH_SNAPPY=OFF
  -DWITH_ZLIB=OFF
  -DWITH_ZSTD=OFF
  -DWITH_LIBURING=${WITH_LIBURING}
  -DWITH_TSAN=${USE_TSAN}
  -DWITH_ASAN=${USE_ASAN}
  -DWITH_UBSAN=${USE_UBSAN}
  -DROCKSDB_BUILD_SHARED=OFF
  -DCMAKE_POSITION_INDEPENDENT_CODE=True
)

if(ROCKSDB_FOUND)
  ExternalProject_Add(rocksdb
    SOURCE_DIR "${RocksDB_ROOT}"
    DOWNLOAD_COMMAND ""
    CMAKE_ARGS ${RocksDB_CMAKE_ARGS}
    BUILD_BYPRODUCTS <BINARY_DIR>/librocksdb.a
    INSTALL_COMMAND ""
  )

  ExternalProject_Get_Property(rocksdb BINARY_DIR)
  set(ROCKSDB_LIBRARIES
      ${BINARY_DIR}/librocksdb.a)
else()
  ExternalProject_Add(rocksdb
    URL https://github.com/facebook/rocksdb/archive/refs/tags/v9.10.0.tar.gz
    CMAKE_ARGS ${RocksDB_CMAKE_ARGS}
    BUILD_BYPRODUCTS <BINARY_DIR>/librocksdb.a
    INSTALL_COMMAND ""
  )

  ExternalProject_Get_Property(rocksdb BINARY_DIR)
  set(ROCKSDB_LIBRARIES
      ${BINARY_DIR}/librocksdb.a)

  ExternalProject_Get_Property(rocksdb SOURCE_DIR)
  set(ROCKSDB_INCLUDE_DIR "${SOURCE_DIR}/include")

  set(ROCKSDB_FOUND TRUE)
endif()

message(STATUS "Found RocksDB library: ${ROCKSDB_LIBRARIES}")
message(STATUS "Found RocksDB includes: ${ROCKSDB_INCLUDE_DIR}")

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

find_library(lz4_STATIC_LIBRARIES
    NAMES liblz4.a REQUIRED)

find_package(Arrow REQUIRED)
find_package(ArrowDataset REQUIRED)

add_executable(server ${SOURCES})
target_sources(server PUBLIC src/table_store.cpp src/encoder.cpp)

target_include_directories(server
		PRIVATE
				${ROCKSDB_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(server
    PRIVATE
        proto
        ${ROCKSDB_LIBRARIES}
				${lz4_STATIC_LIBRARIES}
        Arrow::arrow_shared
				ArrowDataset::arrow_dataset_shared
)
