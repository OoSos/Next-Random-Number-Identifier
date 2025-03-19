# Contributing to Next Random Number Identifier

Thank you for your interest in contributing to this project! Before you begin:

## Understanding the Architecture

Before making changes, please review these important architectural documents:
- [Architecture Documentation](docs/Next%20Random%20Number%20Identifier-architecture-documentation.md)
- [Component Interaction Diagram](docs/diagrams/NRNI%20Component%20Interaction-diagrams.png)
- [Data Flow Diagram](docs/diagrams/NRNI%20Data-flow-diagram.png)
- [Prediction Sequence Diagram](docs/diagrams/NRNI%20Prediction%20sequence-diagram.png)

These documents will help you understand:
- The overall system design
- How components interact
- Data flow through the system
- The prediction process

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

## Documentation Maintenance Process

### Updating Documentation for Architectural Changes

When making significant architectural changes, you MUST update the related documentation to ensure it remains accurate. This is a critical part of the development process.

Follow these steps when making architectural changes:

1. **Identify Affected Documentation**:
   - Review the [Architecture Documentation](docs/Next%20Random%20Number%20Identifier-architecture-documentation.md)
   - Check if any diagrams need updating (Component Interaction, Data Flow, Prediction Sequence)
   - Consider if changes affect interfaces documented in code or markdown files

2. **Update Written Documentation**:
   - Update the Architecture Documentation with your changes
   - Modify code comments and docstrings to reflect new behavior
   - Update any Technical Specifications if required

3. **Update Diagrams**:
   - For diagram changes, edit the `.mermaid` source files in the `docs/diagrams/` directory
   - Use [Mermaid Live Editor](https://mermaid.live/) to preview changes
   - After editing the Mermaid source files, generate updated PNG files for compatibility
   - Ensure both the `.mermaid` and `.png` files are committed together

4. **Documentation Review Checklist**:
   - [ ] Architecture documentation is up-to-date
   - [ ] Diagrams accurately reflect the current architecture
   - [ ] Interface descriptions match implementation
   - [ ] Examples in documentation work with current code
   - [ ] Docstrings are complete and accurate

5. **Include Documentation Changes in Pull Requests**:
   - Documentation updates should be part of the same PR as code changes
   - Reviewers will specifically check documentation for accuracy
   - PRs with significant architectural changes that lack documentation updates will not be merged

### Document-Code Synchronization

Always strive to maintain synchronization between code and documentation:

- If you find documentation that is out of date, create an issue or fix it directly
- When writing new code, document it as you go, not as an afterthought
- Consider documentation as important as the code itself

## Development Process

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Run the tests and linting checks (`pytest` and `flake8`)
4. Commit your changes (`git commit -m 'Add some amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## Development Setup

1. Fork and clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment:
   - Windows: `venv\Scripts\activate`
   - Unix/macOS: `source venv/bin/activate`
4. Install dependencies: `pip install -e ".[dev]"`

## Development Guidelines

- Follow PEP 8 style guide
- Add type hints to all functions
- Write unit tests for new features
- Update documentation when changing functionality
- Run the full test suite before submitting PR

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

## Code Quality Checks

This project uses several tools to maintain code quality:
- Type checking: MyPy
- Code formatting: Black
- Import ordering: isort
- Linting: Flake8
- Testing: pytest

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
- Always update architectural documentation when making structural changes
- Maintain diagram consistency with code changes

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

For PRs with architectural changes, documentation updates will be specifically reviewed for completeness and accuracy.

## Resources

- [GitHub Help](https://help.github.com)
- [Python Testing Documentation](https://docs.python.org/3/library/unittest.html)
- [Mermaid Diagram Syntax](https://mermaid.js.org/syntax/classDiagram.html)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)