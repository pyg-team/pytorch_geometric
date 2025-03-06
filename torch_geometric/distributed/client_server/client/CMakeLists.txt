# Currently, C++ client is only built for testing purposes.
# We write an equivalent client in Python for integration
# with PyG, which does not require linkage.
set(SOURCES
    src/main.cpp
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})
add_executable(client ${SOURCES})
target_link_libraries(client
    PRIVATE
        proto
)

