# Contributing to MLOps Pipeline

Thank you for your interest in contributing to this MLOps project! This document provides guidelines for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/YT_MLOPS_OOPS_TUT.git`
3. Create a virtual environment: `python -m venv venv`
4. Activate the environment: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
5. Install dependencies: `pip install -r requirements-dev.txt`
6. Install the package in editable mode: `pip install -e .`

## Development Workflow

1. Create a new branch for your feature: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Write tests for your changes
4. Run tests: `pytest tests/`
5. Ensure code quality: `flake8 src/`
6. Commit your changes: `git commit -m "Add your feature"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Create a Pull Request

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions small and focused
- Use type hints where applicable

## Testing

- Write unit tests for all new functionality
- Ensure all tests pass before submitting PR
- Aim for high test coverage (>80%)
- Test edge cases and error handling

## Documentation

- Update README.md if adding new features
- Add docstrings to all public APIs
- Include examples for new functionality
- Update configuration documentation if needed

## Commit Messages

- Use clear and descriptive commit messages
- Start with a verb in present tense (Add, Fix, Update, etc.)
- Keep the first line under 50 characters
- Add detailed description if needed

## Pull Request Process

1. Ensure all tests pass
2. Update documentation as needed
3. Add a clear description of your changes
4. Link any related issues
5. Request review from maintainers

## Code Review

- Be open to feedback
- Respond to review comments promptly
- Make requested changes or explain why they're not needed
- Be respectful and constructive

## Questions?

If you have questions, please open an issue for discussion.

Thank you for contributing!
