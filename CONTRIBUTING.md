# Contributing to Custodian Enhanced

Thank you for your interest in contributing to Custodian Enhanced! This document provides guidelines and instructions for contributing to this project.

## 🤝 How to Contribute

### Types of Contributions

We welcome various types of contributions:

- 🐛 **Bug Reports**: Report issues with detailed reproduction steps
- 💡 **Feature Requests**: Suggest new functionality or improvements
- 📖 **Documentation**: Improve guides, documentation, or code comments
- 🧪 **Testing**: Add test cases, improve test coverage, or fix test issues
- 💻 **Code**: Submit bug fixes, new features, or performance improvements
- 🎨 **UI/UX**: Improve user experience and interface design
- 🌍 **Translations**: Add support for additional languages

### Getting Started

1. **Fork the Repository**
   ```bash
   # Fork the repo on GitHub, then clone your fork
   git clone https://github.com/yourusername/custodian-enhanced.git
   cd custodian-enhanced
   ```

2. **Set Up Development Environment**
   ```bash
   # Install dependencies
   pip install -r requirements_enhanced.txt
   
   # Install development dependencies
   pip install pytest black flake8 pre-commit
   
   # Install pre-commit hooks
   pre-commit install
   ```

3. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

## 📋 Development Guidelines

### Code Style

- Follow PEP 8 Python style guidelines
- Use meaningful variable and function names
- Add type hints where appropriate
- Include docstrings for all public functions and classes
- Maximum line length: 88 characters (Black default)

### Code Formatting

We use Black for code formatting:

```bash
# Format your code
black .

# Check formatting
black --check .
```

### Linting

We use flake8 for linting:

```bash
# Run linter
flake8 .
```

### Testing

- Write tests for all new functionality
- Ensure all tests pass before submitting
- Maintain or improve test coverage

```bash
# Run tests
python test_suite.py

# Run validation
python scripts/simple_validation.py

# Generate test documents
python scripts/generate_test_docs.py
```

### Documentation

- Update documentation for any new features
- Include code examples where helpful
- Update README.md if needed
- Add docstrings to new functions and classes

## 🐛 Bug Reports

When reporting bugs, please include:

### Required Information

- **Environment Details**:
  - Python version
  - Operating system
  - GPU/CPU specifications
  - Dependencies versions

- **Bug Description**:
  - Clear description of the issue
  - Expected behavior vs actual behavior
  - Steps to reproduce the problem

- **Supporting Materials**:
  - Error messages and stack traces
  - Log files (sanitized of sensitive data)
  - Sample documents that cause issues (if safe to share)
  - Configuration files (with API keys removed)

### Bug Report Template

```markdown
## Bug Description
Brief description of the issue

## Environment
- Python version: 
- OS: 
- GPU: 
- Dependencies: pip list

## Steps to Reproduce
1. 
2. 
3. 

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Error Messages
```
Paste error messages here
```

## Additional Context
Any other relevant information
```

## 💡 Feature Requests

When requesting features, please include:

- Clear description of the desired functionality
- Use cases and benefits
- Potential implementation approach
- Any related issues or discussions

### Feature Request Template

```markdown
## Feature Description
Brief description of the proposed feature

## Use Case
Why is this feature needed? What problem does it solve?

## Proposed Solution
How should this feature work?

## Alternatives Considered
What other solutions have you considered?

## Additional Context
Any other relevant information
```

## 🔧 Development Process

### Workflow

1. **Check Existing Issues**: Search for existing issues or discussions
2. **Create Issue**: For significant changes, create an issue first
3. **Fork & Branch**: Fork the repo and create a feature branch
4. **Develop**: Implement your changes with tests
5. **Test**: Ensure all tests pass
6. **Document**: Update documentation as needed
7. **Submit**: Create a pull request

### Pull Request Guidelines

#### Before Submitting

- [ ] Code follows project style guidelines
- [ ] All tests pass (`python test_suite.py`)
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] Commit messages are clear and descriptive
- [ ] Changes are focused and atomic

#### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or marked as such)
```

### Review Process

1. **Automatic Checks**: CI/CD runs automatically
2. **Code Review**: Maintainers review the code
3. **Feedback**: Address any requested changes
4. **Approval**: Once approved, changes will be merged

## 🎯 Coding Standards

### Python Specifics

```python
# Good: Clear function with type hints and docstring
def process_document(file_path: str, confidence_threshold: float = 0.8) -> Dict[str, Any]:
    """Process a single document and return analysis results.
    
    Args:
        file_path: Path to the document file
        confidence_threshold: Minimum confidence score for processing
        
    Returns:
        Dictionary containing processing results and metadata
        
    Raises:
        FileNotFoundError: If the specified file doesn't exist
        ProcessingError: If document processing fails
    """
    # Implementation here
    pass

# Bad: Unclear function without documentation
def proc_doc(fp, ct=0.8):
    # Implementation here
    pass
```

### Error Handling

- Use specific exception types
- Provide helpful error messages
- Log errors appropriately
- Handle edge cases gracefully

```python
# Good: Specific exception handling
try:
    result = process_document(file_path)
except FileNotFoundError:
    logger.error(f"Document not found: {file_path}")
    return None
except ProcessingError as e:
    logger.error(f"Processing failed for {file_path}: {e}")
    return None
```

### Logging

- Use the existing logging infrastructure
- Include context in log messages
- Use appropriate log levels

```python
# Good: Descriptive logging
logger.info(f"Processing document: {os.path.basename(file_path)}")
logger.debug(f"Using confidence threshold: {confidence_threshold}")
logger.error(f"Failed to process {file_path}: {error_message}")
```

## 🧪 Testing Guidelines

### Test Structure

- Place tests in appropriate test files
- Use descriptive test names
- Include both positive and negative test cases
- Test edge cases and error conditions

```python
def test_document_processing_success():
    """Test successful document processing with valid input."""
    # Test implementation
    pass

def test_document_processing_invalid_file():
    """Test document processing with invalid file path."""
    # Test implementation
    pass

def test_document_processing_corrupted_file():
    """Test document processing with corrupted file."""
    # Test implementation
    pass
```

### Test Data

- Use synthetic test data only
- Never include real personal or sensitive information
- Generate test documents programmatically when possible

## 📚 Documentation Standards

### Code Documentation

- Add docstrings to all public functions and classes
- Include parameter descriptions and return values
- Document exceptions that may be raised
- Provide usage examples for complex functions

### External Documentation

- Update relevant documentation files
- Include screenshots for UI changes
- Provide clear setup instructions
- Add troubleshooting information

## 🚀 Release Process

### Version Numbering

We follow Semantic Versioning (SemVer):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Changelog

- Document all changes in CHANGELOG.md
- Categorize changes (Added, Changed, Deprecated, Removed, Fixed, Security)
- Include migration guides for breaking changes

## 📞 Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Pull Request Comments**: Code-specific discussions

### Questions?

If you have questions about contributing:

1. Check existing documentation
2. Search GitHub issues and discussions
3. Create a new discussion for general questions
4. Create an issue for specific bugs or features

## 🙏 Recognition

Contributors will be recognized in:
- Project README
- Release notes
- Contributors section

Thank you for helping make Custodian Enhanced better! 🎉