cmake_minimum_required(VERSION 3.16)
project(HFTMarketMaker VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
endif()

# Enable position independent code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Find packages
find_package(Threads REQUIRED)
find_package(Boost 1.70 REQUIRED COMPONENTS system date_time thread filesystem)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Find QuickFIX
find_path(QUICKFIX_INCLUDE_DIR quickfix/Application.h
    HINTS /usr/local/include /usr/include
)
find_library(QUICKFIX_LIBRARY quickfix
    HINTS /usr/local/lib /usr/lib
)

# Find WebSocket++
find_path(WEBSOCKETPP_INCLUDE_DIR websocketpp/config/asio_client.hpp
    HINTS /usr/local/include /usr/include
)

# Find nlohmann/json
find_package(nlohmann_json 3.2.0 REQUIRED)

# Find OpenSSL for WebSocket TLS
find_package(OpenSSL REQUIRED)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${Boost_INCLUDE_DIRS}
    ${QUICKFIX_INCLUDE_DIR}
    ${WEBSOCKETPP_INCLUDE_DIR}
    ${Python3_INCLUDE_DIRS}
)

# Core library
add_library(hft_core STATIC
    src/core/order_book.cpp
    src/core/fix_handler.cpp
)

target_link_libraries(hft_core
    ${CMAKE_THREAD_LIBS_INIT}
    ${Boost_LIBRARIES}
    ${QUICKFIX_LIBRARY}
)

# Strategies library
add_library(hft_strategies STATIC
    src/strategies/avellaneda_stoikov.cpp
)

target_link_libraries(hft_strategies
    hft_core
)

# Models library (with Python integration)
add_library(hft_models STATIC
    src/models/python_bridge.cpp
)

target_link_libraries(hft_models
    hft_core
    ${Python3_LIBRARIES}
)

# Set Python paths for models library
target_compile_definitions(hft_models PRIVATE
    PYTHON_MODULE_PATH="${CMAKE_SOURCE_DIR}/python/models"
)

# Data feed library
add_library(hft_data STATIC
    src/data/polygon_websocket_feed.cpp
)

target_link_libraries(hft_data
    hft_core
    ${Boost_LIBRARIES}
    OpenSSL::SSL
    OpenSSL::Crypto
    nlohmann_json::nlohmann_json
)

# Risk management library
add_library(hft_risk STATIC
    src/risk/risk_manager.cpp
)

target_link_libraries(hft_risk
    hft_core
    hft_strategies
)

# Main market maker executable
add_executable(market_maker
    src/main.cpp
)

target_link_libraries(market_maker
    hft_core
    hft_strategies
    hft_models
    hft_data
    hft_risk
    ${CMAKE_THREAD_LIBS_INIT}
)

# Backtesting executable
add_executable(backtest
    src/backtest.cpp
)

target_link_libraries(backtest
    hft_core
    hft_strategies
    hft_models
    hft_data
    hft_risk
)

# Tests
enable_testing()

# Order book tests
add_executable(test_order_book
    tests/test_order_book.cpp
)

target_link_libraries(test_order_book
    hft_core
)

add_test(NAME test_order_book COMMAND test_order_book)

# Strategy tests
add_executable(test_strategies
    tests/test_strategies.cpp
)

target_link_libraries(test_strategies
    hft_strategies
    hft_core
)

add_test(NAME test_strategies COMMAND test_strategies)

# Risk management tests
add_executable(test_risk
    tests/test_risk.cpp
)

target_link_libraries(test_risk
    hft_risk
    hft_core
)

add_test(NAME test_risk COMMAND test_risk)

# Performance benchmarks
add_executable(benchmark_order_book
    benchmarks/benchmark_order_book.cpp
)

target_link_libraries(benchmark_order_book
    hft_core
)

# Installation rules
install(TARGETS market_maker backtest
    RUNTIME DESTINATION bin
)

install(TARGETS hft_core hft_strategies hft_models hft_data hft_risk
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include
)

install(DIRECTORY python/
    DESTINATION share/hft_market_maker/python
)

install(DIRECTORY config/
    DESTINATION etc/hft_market_maker
)

# Package configuration
include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/HFTMarketMakerConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

# Print configuration summary
message(STATUS "=== HFT Market Maker Configuration ===")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "QuickFIX: ${QUICKFIX_INCLUDE_DIR}")
message(STATUS "WebSocket++: ${WEBSOCKETPP_INCLUDE_DIR}")
message(STATUS "Python: ${Python3_VERSION}")
message(STATUS "Boost: ${Boost_VERSION}")
message(STATUS "======================================")

# CPack configuration for packaging
set(CPACK_PACKAGE_NAME "HFTMarketMaker")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "High-Frequency Market Making System")
set(CPACK_PACKAGE_VENDOR "HFT Systems")
set(CPACK_PACKAGE_CONTACT "support@hftsystems.com")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/README.md")

# Debian package specific
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libboost-all-dev (>= 1.70), python3 (>= 3.8)")
set(CPACK_DEBIAN_PACKAGE_SECTION "finance")

include(CPack)