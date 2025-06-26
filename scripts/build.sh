#!/bin/bash

# Build script for HFT Market Maker

set -e  # Exit on error

echo "Building HFT Market Maker..."

# Check dependencies
echo "Checking dependencies..."

if ! command -v cmake &> /dev/null; then
    echo "ERROR: cmake not found. Please install CMake 3.16+"
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found. Please install Python 3.8+"
    exit 1
fi

# Create build directory
BUILD_DIR="build"
if [ -d "$BUILD_DIR" ]; then
    echo "Cleaning existing build directory..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake
echo "Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
echo "Building..."
make -j$(nproc || sysctl -n hw.ncpu || echo 4)

# Run tests
echo "Running tests..."
make test

# Install Python dependencies
echo "Installing Python dependencies..."
cd ../python
pip3 install -r requirements.txt

cd ..

echo "Build complete!"
echo ""
echo "To run the market maker:"
echo "  ./build/bin/market_maker config/config.json"
echo ""
echo "To run backtesting:"
echo "  ./build/bin/backtest --config config/backtest.json"