# Contributing to FungiSense AI

Thank you for your interest in contributing to FungiSense AI! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of background or experience level.

### Expected Behavior

- Be respectful and professional
- Welcome newcomers and help them get started
- Accept constructive criticism gracefully
- Focus on what's best for the project and community

### Unacceptable Behavior

- Harassment, discrimination, or offensive comments
- Trolling or personal attacks
- Publishing others' private information
- Any conduct that would be inappropriate in a professional setting

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of FastAPI and TensorFlow
- Familiarity with mushroom classification concepts (helpful but not required)

### Setting Up Your Development Environment

1. **Fork the repository**
   ```bash
   # Click 'Fork' on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/fungisense-ai.git
   cd fungisense-ai
   ```

2. **Add upstream remote**
   ```bash
   git remote add upstream https://github.com/ORIGINAL_OWNER/fungisense-ai.git
   ```

3. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Verify installation**
   ```bash
   pytest tests/ -v
   python api/main.py  # Should start the server
   ```

## Development Workflow

### 1. Create a Branch

```bash
# Update your local main branch
git checkout main
git pull upstream main

# Create a feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

### Branch Naming Conventions

- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `test/` - Test additions or modifications
- `refactor/` - Code refactoring
- `perf/` - Performance improvements

### 2. Make Your Changes

- Write clear, self-documenting code
- Add comments for complex logic
- Follow existing code patterns
- Keep changes focused and atomic

### 3. Test Your Changes

```bash
# Run tests
pytest tests/ -v

# Run specific test file
pytest tests/test_endpoints.py -v

# Check code coverage
pytest --cov=api --cov=src tests/

# Format code
black api/ src/ scripts/

# Check linting
flake8 api/ src/ scripts/ --max-line-length=100
```

### 4. Commit Your Changes

Follow [Conventional Commits](https://www.conventionalcommits.org/) format:

```bash
# Format: <type>(<scope>): <subject>

git add .
git commit -m "feat: add species comparison endpoint"
git commit -m "fix: correct edibility risk calculation"
git commit -m "docs: update API documentation"
git commit -m "test: add tests for batch prediction"
```

**Commit Types:**
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation only
- `style:` - Code style (formatting, missing semicolons, etc.)
- `refactor:` - Code refactoring
- `perf:` - Performance improvement
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Coding Standards

### Python Style Guide

We follow **PEP 8** with these specific guidelines:

- **Line length**: Maximum 100 characters
- **Formatter**: Use Black (automatic formatting)
- **Imports**: Organized (standard library, third-party, local)
- **Type hints**: Required for all function parameters and return values
- **Docstrings**: Required for all public functions, classes, and modules

### Code Formatting

```bash
# Format all code
black api/ src/ scripts/

# Check without making changes
black --check api/ src/ scripts/
```

### Linting

```bash
# Run flake8
flake8 api/ src/ scripts/ --max-line-length=100

# Ignore specific rules (if needed)
flake8 api/ --ignore=E501,W503
```

### Type Hints Example

```python
from typing import List, Dict, Optional

def predict_species(
    features: Dict[str, any],
    confidence_threshold: float = 0.5
) -> Dict[str, any]:
    """
    Predict mushroom species from features.
    
    Args:
        features: Dictionary of mushroom characteristics
        confidence_threshold: Minimum confidence for prediction
        
    Returns:
        Dictionary containing prediction and confidence score
        
    Raises:
        ValueError: If features are invalid
    """
    # Implementation
    pass
```

### Docstring Format

Use Google-style docstrings:

```python
def function_name(param1: str, param2: int) -> bool:
    """
    Brief description of function.
    
    Longer description if needed. Explain what the function does,
    not how it does it (the code shows that).
    
    Args:
        param1: Description of first parameter
        param2: Description of second parameter
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When and why this is raised
        TypeError: When and why this is raised
        
    Example:
        >>> function_name("test", 42)
        True
    """
    pass
```

## Testing

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use descriptive test names that explain what is being tested
- Follow the Arrange-Act-Assert pattern

### Test Example

```python
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_complete_prediction_success():
    """Test successful complete prediction with valid data."""
    # Arrange
    mushroom_data = {
        "cap_diameter": 12.5,
        "cap_shape": "convex",
        # ... all required fields
    }
    
    # Act
    response = client.post("/api/v1/predict/complete", json=mushroom_data)
    
    # Assert
    assert response.status_code == 200
    result = response.json()
    assert "species" in result
    assert "edibility" in result
    assert result["species"]["confidence"] > 0
```

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_endpoints.py

# Run specific test function
pytest tests/test_endpoints.py::test_complete_prediction_success

# Run with coverage
pytest --cov=api --cov=src tests/

# Generate HTML coverage report
pytest --cov=api --cov=src --cov-report=html tests/
open htmlcov/index.html
```

## Documentation

### API Documentation

- FastAPI generates automatic docs from code
- Add clear descriptions to Pydantic models
- Use docstrings in route functions
- Update PHASE1_ENDPOINTS.md for major changes

### Code Documentation

- Add docstrings to all public functions and classes
- Comment complex algorithms or business logic
- Keep comments up-to-date with code changes
- Avoid obvious comments

### README Updates

When adding features, update:
- Feature list in README.md
- API endpoints section
- Usage examples if applicable
- Installation steps if dependencies change

## Pull Request Process

### Before Submitting

- [ ] Code follows style guidelines (run `black` and `flake8`)
- [ ] All tests pass (`pytest`)
- [ ] New tests added for new functionality
- [ ] Documentation updated (README, docstrings, etc.)
- [ ] Commit messages follow Conventional Commits
- [ ] Branch is up-to-date with main

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
Describe the tests you ran and how to reproduce

## Checklist
- [ ] My code follows the style guidelines
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have updated the documentation accordingly
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
```

### Review Process

1. Automated checks must pass (if configured)
2. At least one maintainer review required
3. Address review comments
4. Maintainer will merge when approved

## Issue Reporting

### Bug Reports

Use this template:

```markdown
**Description**
Clear description of the bug

**To Reproduce**
Steps to reproduce:
1. Go to '...'
2. Click on '...'
3. See error

**Expected Behavior**
What you expected to happen

**Actual Behavior**
What actually happened

**Environment**
- OS: [e.g., Windows 10, Ubuntu 20.04]
- Python version: [e.g., 3.10.5]
- Package versions: [output of `pip freeze`]

**Additional Context**
Any other context about the problem

**Screenshots**
If applicable
```

### Feature Requests

```markdown
**Is your feature request related to a problem?**
A clear description of the problem

**Describe the solution you'd like**
A clear description of what you want to happen

**Describe alternatives you've considered**
Other solutions you've thought about

**Additional context**
Any other context or screenshots
```

## Areas Needing Contribution

We especially welcome contributions in these areas:

- 🧪 **Testing**: Increase test coverage
- 📝 **Documentation**: Improve guides and examples
- 🎨 **Frontend**: Build web interface
- 📱 **Mobile**: iOS/Android apps
- 🌍 **Internationalization**: Multi-language support
- 📸 **Image Classification**: CNN-based visual identification
- 🗺️ **Geographic Data**: Region-specific species
- 🔬 **Model Improvements**: Better accuracy and performance
- 🐳 **DevOps**: CI/CD, deployment guides

## Questions?

- Check existing issues and discussions
- Ask in issue comments
- Reach out to maintainers

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to FungiSense AI! 🍄

