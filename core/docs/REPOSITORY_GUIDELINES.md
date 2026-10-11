# CHiPPY Repository Guidelines

Guidelines for maintaining and contributing to the CHiPPY pipeline repository.

## Repository Structure Standards

### Directory Organization
```
CHiPPY/
├── 0X_stagename/         # Numbered stages in pipeline order
│   ├── README.md         # Stage-specific documentation (REQUIRED)
│   ├── main_script.py    # Entry point(s)
│   ├── config.py         # Configuration file (if needed)
│   ├── submodules/       # Supporting modules
│   ├── input_data/       # Input directory (.gitkeep if empty)
│   └── output/           # Output directory (.gitkeep if empty)
├── docs/                 # Documentation
├── tests/               # Test suite (if applicable)
├── requirements.txt     # MUST exist at root
└── README.md           # MUST exist at root
```

### Naming Conventions
- **Directories**: lowercase with underscores (e.g., `stage1_image_prep`)
- **Files**: snake_case (e.g., `config_loader.py`)
- **Classes**: PascalCase (e.g., `PDFProcessor`)
- **Constants**: UPPER_CASE (e.g., `MAX_TOKENS`)

## Code Standards

### Python Style
- Follow PEP 8
- Use type hints where practical
- Maximum line length: 100 characters
- Use docstrings for all functions and classes

### Example Module Structure
```python
"""
Module description.
Primary use case and key functions.
"""

from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    """Main processor class."""
    
    def __init__(self, config: Dict):
        """Initialize processor with configuration."""
        self.config = config
    
    def process(self, data: List[str]) -> List[str]:
        """Process input data.
        
        Args:
            data: Input data to process
            
        Returns:
            Processed data
        """
        return [item.process() for item in data]

def helper_function(param: str) -> str:
    """Helper function description."""
    return param.upper()
```

## Documentation Standards

### README Format (Per Stage)
Each stage MUST have:
1. Overview/Purpose
2. Quick Start section
3. Configuration options
4. Usage examples
5. Troubleshooting
6. Integration with next stage

### Code Comments
- Comment WHY, not WHAT
- Explain non-obvious logic
- Mark technical debt with `TODO:` and `FIXME:`

### Configuration Files
- All config files MUST have comments
- Group related settings
- Provide reasonable defaults

## Git Workflow

### Commit Messages
```
Type: Brief description (50 chars max)

Longer explanation if needed.
- Explain what changed
- Explain why
- Note any breaking changes

Fixes #issue-number
```

**Types**:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `refactor:` Code restructuring
- `test:` Test additions
- `perf:` Performance improvement
- `chore:` Maintenance

### Branch Naming
- `feature/description` - New features
- `fix/issue-number` - Bug fixes
- `docs/topic` - Documentation
- `refactor/component` - Code improvements

### Pull Request Template
```md
## Changes
Describe what changed.

## Testing
How was this tested?

## Checklist
- [ ] Code follows style guide
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] No breaking changes
```

## Testing Standards

### Test Organization
```
tests/
├── __init__.py
├── test_stage1.py       # Tests for stage 1
├── test_stage2.py       # Tests for stage 2
└── fixtures/            # Test data
    ├── sample.pdf
    └── expected_output.txt
```

### Test Template
```python
import pytest
from stage1 import preprocess

class TestPreprocess:
    """Test preprocessing functions."""
    
    def test_valid_input(self):
        """Test with valid input."""
        result = preprocess("sample.pdf")
        assert result is not None
        assert len(result) > 0
    
    def test_invalid_input(self):
        """Test error handling."""
        with pytest.raises(ValueError):
            preprocess("nonexistent.pdf")

def test_integration():
    """Integration test."""
    # Test full pipeline
    pass

@pytest.fixture
def sample_data():
    """Provide test data."""
    return {"key": "value"}
```

### Running Tests
```bash
pytest tests/                    # Run all tests
pytest tests/test_stage1.py     # Run specific test
pytest -v                        # Verbose output
pytest --cov                     # Coverage report
```

## Configuration Management

### Config File Structure
```python
"""
Configuration for Stage X

All settings have defaults. Override in environment-specific config.
"""

# Image Processing (Stage 1)
DPI = 200                    # Resolution: 100-300
BRIGHTNESS = 127             # For binary: 0-255
DENOISE_STRENGTH = 1.5      # Strength: 0.5-3.0

# Optional: Environment-specific overrides
import os
if os.getenv('ENVIRONMENT') == 'production':
    DPI = 150  # Lower res for speed
```

### Environment Variables
```bash
# .env file (DO NOT COMMIT)
FLASK_ENV=development
DEBUG=true
API_KEY=<your-key>
DATABASE_URL=postgresql://...
```

## Dependency Management

### Adding Dependencies
1. Add to `requirements.txt` with version: `package>=1.2.0`
2. Document why it's needed (comment)
3. Test complete installation
4. Update documentation if needed

### Optional Dependencies
```python
# In requirements.txt
torch>=2.0.0  # Optional: for GPU acceleration
# Installation: pip install -r requirements.txt
# For GPU: pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Documentation Maintenance

### Update Checklist
- [ ] README.md reflects current functionality
- [ ] All config options documented
- [ ] Troubleshooting covers known issues
- [ ] Examples are current and working
- [ ] Links are not broken
- [ ] Terminology is consistent

### Documentation Tools
- Markdown for all docs (.md)
- Code examples must be tested
- API documentation via docstrings
- Architecture diagrams via ASCII or Mermaid

## Performance Guidelines

### Profiling
```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()
# Your code here
profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)
```

### Benchmarking
```python
import time

start = time.time()
result = function_to_test()
elapsed = time.time() - start
print(f"Took {elapsed} seconds")
```

## Release Process

### Version Numbering
Use semantic versioning: `MAJOR.MINOR.PATCH`
- `1.0.0` - Initial release
- `1.1.0` - New feature
- `1.0.1` - Bug fix
- `2.0.0` - Breaking changes

### Release Steps
1. Update version in relevant files
2. Update CHANGELOG.md
3. Tag release: `git tag v1.0.0`
4. Push: `git push origin v1.0.0`
5. Create GitHub release (if using GitHub)

## Code Review Checklist

- [ ] Code follows style guide (PEP 8)
- [ ] Has appropriate docstrings
- [ ] Includes tests or updates existing ones
- [ ] Documentation updated
- [ ] No console.log or debug statements
- [ ] No hardcoded paths or credentials
- [ ] Performance acceptable
- [ ] Breaking changes documented
- [ ] Backwards compatible (unless major version)

## Troubleshooting Common Issues

### Import Errors
```python
# Use absolute imports
from package.submodule import function  # Good
from ..module import function          # Avoid
```

### Circular Dependencies
```python
# Solution: Import inside function if needed
def function():
    from other_module import dependency
    return dependency
```

### Performance Issues
1. Profile first: `cProfile`
2. Identify bottleneck
3. Add caching if appropriate
4. Consider parallel processing
5. Benchmark improvement

## Maintenance Tasks

### Regular (Weekly)
- [ ] Review new issues
- [ ] Respond to questions
- [ ] Check test coverage

### Monthly
- [ ] Review dependencies for updates
- [ ] Update security patches
- [ ] Review performance metrics
- [ ] Documentation review

### Quarterly
- [ ] Major dependency updates
- [ ] Performance optimization review
- [ ] Code cleanup
- [ ] Architecture review

## Security Guidelines

### Never Commit
- Credentials, API keys
- Passwords
- Private configuration
- Large data files

### Use Instead
- `.env` files (in .gitignore)
- Environment variables
- Secrets management systems
- Configuration servers

### Input Validation
```python
def process_user_input(user_input: str) -> str:
    """Validate and sanitize user input."""
    if not isinstance(user_input, str):
        raise TypeError("Input must be string")
    if len(user_input) > 1000:
        raise ValueError("Input too long")
    # Additional validation
    return sanitize(user_input)
```

## Contribution Categories

### Welcome Contributions
- Bug fixes
- Performance improvements
- Documentation enhancements
- New OCR backends
- Additional language support
- Test coverage
- Examples and tutorials

### Needs Discussion First
- New stages
- Major refactoring
- New dependencies
- Breaking changes
- Performance tradeoffs

## Getting Help

1. Check documentation
2. Search existing issues
3. File issue with:
   - Problem description
   - Minimal reproduction
   - Environment info
   - Error messages
4. For questions: Use discussions (if available)

## Attribution

All contributions must be attributed:
- Commits: Author in git history
- Special features: Acknowledge in CONTRIBUTORS.md
- Large contributions: Add to project metadata

---

**Last Updated**: March 2026
**Maintained by**: CHiPPY Team
