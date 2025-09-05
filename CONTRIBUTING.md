# Contributing to SQLVector

First off, thank you for considering contributing to SQLVector! It's people like you that make SQLVector such a great tool.

## Code of Conduct

By participating in this project, you are expected to uphold our Code of Conduct:

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps which reproduce the problem**
- **Provide specific examples to demonstrate the steps**
- **Describe the behavior you observed after following the steps**
- **Explain which behavior you expected to see instead and why**
- **Include Python version, SQLVector version, and backend being used**

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

- **Use a clear and descriptive title**
- **Provide a step-by-step description of the suggested enhancement**
- **Provide specific examples to demonstrate the steps**
- **Describe the current behavior and explain which behavior you expected to see instead**
- **Explain why this enhancement would be useful**

### Pull Requests

1. Fork the repo and create your branch from `main`
2. If you've added code that should be tested, add tests
3. If you've changed APIs, update the documentation
4. Ensure the test suite passes
5. Make sure your code follows the existing style
6. Issue that pull request!

## Development Setup

### Prerequisites

- Python 3.12 or higher
- Git
- Optional: `uv` for faster package management

### Setting Up Your Development Environment

1. **Fork and clone the repository**

```bash
git clone https://github.com/your-username/sqlvector.git
cd sqlvector
```

2. **Create a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**

```bash
# Install with all optional dependencies for development
pip install -e ".[duckdb,test]"

# Or using uv (faster)
uv pip install -e ".[duckdb,test]"
```

4. **Install pre-commit hooks (optional but recommended)**

```bash
pip install pre-commit
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=sqlvector tests/

# Run specific test file
pytest tests/test_duckdb.py

# Run tests with verbose output
pytest -v

# Run tests matching a pattern
pytest -k "test_load_documents"
```

### Testing Different Backends

```bash
# Test DuckDB backend
pytest tests/test_duckdb.py tests/test_duckdb_vss.py

# Test SQLite backend
pytest tests/test_sqlite.py tests/test_sqlite_vss.py

# Test core functionality
pytest tests/test_embedding.py tests/test_loader.py tests/test_query.py
```

## Code Style Guidelines

### Python Style

- Follow [PEP 8](https://pep8.org/)
- Use type hints for all function signatures
- Keep functions focused and small
- Write docstrings for all public functions and classes
- Use meaningful variable names

### Example Code Style

```python
from typing import List, Dict, Any, Optional

async def load_documents(
    self,
    documents: List[Dict[str, Any]],
    batch_size: Optional[int] = None
) -> List[str]:
    """
    Load documents into the database with embeddings.
    
    Args:
        documents: List of documents with 'content' and optional 'metadata'
        batch_size: Number of documents to process in each batch
        
    Returns:
        List of document IDs
        
    Raises:
        ValueError: If documents are not in the correct format
    """
    # Implementation here
    pass
```

### Import Organization

Organize imports in the following order:
1. Standard library imports
2. Related third party imports
3. Local application/library specific imports

Use absolute imports for clarity.

### Testing Guidelines

- Write tests for all new functionality
- Maintain or increase code coverage
- Use descriptive test names that explain what is being tested
- Group related tests in test classes
- Use fixtures for common test setup
- Mock external dependencies appropriately

### Documentation

- Update README.md if adding new features
- Add docstrings to all new functions and classes
- Update type hints for better IDE support
- Add usage examples for new functionality
- Keep documentation up to date with code changes

## Project Structure

```
sqlvector/
├── sqlvector/              # Main package
│   ├── __init__.py       # Package initialization and exports
│   ├── config.py         # Configuration classes
│   ├── embedding.py      # Embedding providers
│   ├── loader.py         # Document loader
│   ├── query.py          # Query engine
│   ├── models.py         # SQLAlchemy models
│   ├── protocols.py      # Protocol definitions
│   └── backends/         # Backend implementations
│       ├── duckdb/       # DuckDB backend
│       └── sqlite/       # SQLite backend
├── tests/                # Test suite
├── examples/             # Example scripts
├── docs/                 # Additional documentation
└── pyproject.toml        # Project configuration
```

## Making Changes

### Backend Development

When adding a new backend:

1. Create a new directory under `sqlvector/backends/`
2. Implement the required protocols from `sqlvector/protocols.py`
3. Add configuration class extending `DatabaseConfig`
4. Add comprehensive tests in `tests/test_<backend>.py`
5. Update README.md with backend-specific documentation

### Adding Features

1. Discuss the feature in an issue first
2. Implement the feature with tests
3. Update documentation
4. Submit a pull request with a clear description

## Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line

### Examples

```
Add DuckDB HNSW indexing support

- Implement HNSW index creation and management
- Add configuration options for HNSW parameters
- Include benchmarks comparing with flat search

Fixes #123
```

## Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create a git tag: `git tag -a v0.1.0 -m "Version 0.1.0"`
4. Push tag: `git push origin v0.1.0`
5. GitHub Actions will handle PyPI deployment

## Community

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussions
- **Pull Requests**: For contributing code changes

## Questions?

Feel free to open an issue with your question or reach out through GitHub Discussions.

Thank you for contributing to SQLVector! 🎉