# Setup & Build Guide

## C++ Build

```bash
# Build
mkdir build && cd build
cmake ..
cmake --build .

# Run tests
./kdtree_test                   # All tests including benchmarks
./kdtree_test --skip-benchmarks # Skip benchmarks
./kdtree_test '[benchmark]'     # Only benchmarks
```

## Python build in a Virtual Environment

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python package
pip install -e .

# Run examples
python example.py

# Run tests
pytest kdtree_test.py -v

# When done
deactivate
```

## Git Hooks Setup

```bash
# Copy hooks (they auto-detect and use venv if available)
cp pre-commit pre-push .git/hooks/
```

Hooks will:
- pre-commit: Run fast tests (no benchmarks)
- pre-push: Run full suite including benchmarks

## Troubleshooting

### Python module not found after install
```bash
# Make sure you're in the venv
source venv/bin/activate

# Verify install
pip list | grep kdtree

# Reinstall if needed
pip install -e . --force-reinstall
```

### C++ compilation errors
```bash
# Make sure you have C++20 compiler
g++ --version  # Should be 10+ for GCC
clang++ --version  # Should be 12+ for Clang

# Clean build
rm -rf build && mkdir build && cd build && cmake .. && cmake --build .
```

### Git hooks not running
```bash
# Make them executable
chmod +x .git/hooks/pre-commit .git/hooks/pre-push

# Test manually
.git/hooks/pre-commit
```

## IDE Setup

### VSCode
```json
// .vscode/settings.json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
  "python.testing.pytestEnabled": true,
  "C_Cpp.default.cppStandard": "c++20",
  "C_Cpp.default.includePath": ["${workspaceFolder}"]
}
```

## Dependencies

### C++
- C++20 compiler (GCC 10+, Clang 12+, MSVC 2019+)
- CMake 3.14+

### Python
- Python 3.8+
- pybind11 (auto-installed)
- pytest (for testing)
