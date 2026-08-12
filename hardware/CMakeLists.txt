cmake_minimum_required(VERSION 3.16)

project(gdn_hls CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -Wno-unknown-pragmas -mcmodel=large)
add_compile_definitions(USE_CPU_ONLY)

add_executable(gdn_host
  src/decode.cpp
  src/main.cpp
  src/vocab.cpp
  src/weight.cpp
)
target_include_directories(gdn_host PRIVATE src)
