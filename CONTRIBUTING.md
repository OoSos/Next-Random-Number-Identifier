# Contributing to Next Random Number Identifier

Thank you for considering contributing to the Next Random Number Identifier project! This document outlines the process and guidelines for contributing.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone.

## How Can I Contribute?

### Reporting Bugs

This section guides you through submitting a bug report. Following these guidelines helps maintainers understand your report, reproduce the behavior, and find related reports.

- Use the bug report template provided
- Be as detailed as possible
- Include steps to reproduce the issue
- Describe the expected vs. actual behavior
- Include relevant environment information

### Suggesting Enhancements

This section guides you through submitting an enhancement suggestion, including completely new features and minor improvements to existing functionality.

- Use the feature request template provided
- Provide a clear description of the enhancement
- Describe the current behavior and explain which behavior you expected to see instead
- Explain why this enhancement would be useful

### Pull Requests

- Fill in the required template
- Follow the code style guidelines
- Include adequate tests for your changes
- Update any relevant documentation
- The PR should work for Python 3.8, 3.9, and 3.10

## Development Setup

1. Fork and clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment:
   - Windows: `venv\Scripts\activate`
   - Unix/macOS: `source venv/bin/activate`
4. Install dependencies: `pip install -e ".[dev]"`

## Project Structure

- `src/`: Source code files
- `tests/`: Test files
- `data/`: Data files and datasets
- `docs/`: Documentation

## Testing

Run tests with pytest:

```bash
pytest
```

For coverage report:

```bash
pytest --cov=src tests/
```

## Code Style

This project follows these coding guidelines:

- Use [Black](https://black.readthedocs.io/) for code formatting
- Sort imports with [isort](https://pycqa.github.io/isort/)
- Follow [PEP 484](https://www.python.org/dev/peps/pep-0484/) for type hints
- Lint with [flake8](https://flake8.pycqa.org/)

Automated style checking:

```bash
black .
isort .
flake8
mypy src
```

## Documentation

- Follow Google-style docstrings
- Update documentation for new features
- Include type hints in function signatures

## Commit Messages

- Use clear and descriptive commit messages
- Reference issues and pull requests when relevant
- Keep commits focused on a single change

## Review Process

All submissions require review. We use GitHub pull requests for this purpose.

1. Submit your pull request
2. Respond to any feedback or questions
3. Make any requested changes
4. Once approved, a maintainer will merge your changes

## Resources

- [GitHub Help](https://help.github.com)
- [Python Testing Documentation](https://docs.python.org/3/library/unittest.html)