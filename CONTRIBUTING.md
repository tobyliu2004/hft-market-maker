# Contributing to HFT Market Maker

Thank you for your interest in contributing to the HFT Market Maker project! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please note that this project adheres to a Code of Conduct. By participating, you are expected to uphold this code.

## How to Contribute

### Reporting Issues

- Check if the issue has already been reported
- Provide a clear and descriptive title
- Include steps to reproduce the issue
- Specify your environment (OS, compiler version, dependencies)
- Include relevant log outputs or error messages

### Submitting Pull Requests

1. **Fork the Repository**
   - Fork the project on GitHub
   - Clone your fork locally

2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Your Changes**
   - Follow the coding standards below
   - Add tests for new functionality
   - Update documentation as needed

4. **Commit Your Changes**
   - Use clear and meaningful commit messages
   - Follow conventional commit format: `type(scope): description`
   - Examples:
     - `feat(strategy): add momentum-based market making strategy`
     - `fix(risk): correct position limit calculation`
     - `docs(readme): update installation instructions`

5. **Run Tests**
   ```bash
   cd build
   make test
   ```

6. **Push to Your Fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**
   - Provide a clear description of the changes
   - Reference any related issues
   - Ensure all CI checks pass

## Coding Standards

### C++ Style Guide

- Follow modern C++20 standards
- Use meaningful variable and function names
- Keep functions focused and under 50 lines when possible
- Use RAII and smart pointers for resource management

### Code Organization

```
include/      # Header files
  core/       # Core system components
  data/       # Market data handling
  models/     # Statistical and ML models
  risk/       # Risk management
  strategies/ # Trading strategies
  utils/      # Utilities

src/          # Implementation files
  (mirrors include structure)

tests/        # Unit and integration tests
python/       # Python components
```

### Testing

- Write unit tests for all new functionality
- Maintain test coverage above 80%
- Use Google Test framework for C++ tests
- Use pytest for Python tests

### Documentation

- Document all public APIs using Doxygen format
- Include examples in documentation
- Update README.md for significant changes
- Add inline comments for complex algorithms

## Development Setup

### Prerequisites

- C++20 compatible compiler (GCC 10+, Clang 12+)
- CMake 3.16+
- Boost 1.70+
- Python 3.8+
- QuickFIX 1.15+

### Building

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j8
```

### Running Tests

```bash
make test
# Or for verbose output:
ctest -V
```

### Code Analysis

```bash
# Static analysis
make cppcheck

# Code formatting
find . -name "*.cpp" -o -name "*.h" | xargs clang-format -i
```

## Performance Considerations

- Minimize allocations in hot paths
- Use lock-free data structures where appropriate
- Profile before optimizing
- Document performance-critical sections

## Security

- Never commit API keys or credentials
- Validate all external inputs
- Use secure communication protocols
- Follow principle of least privilege

## Review Process

1. Code review by at least one maintainer
2. All tests must pass
3. No decrease in code coverage
4. Documentation updated
5. Performance benchmarks show no regression

## Questions?

Feel free to open an issue for any questions about contributing.

Thank you for contributing to HFT Market Maker!