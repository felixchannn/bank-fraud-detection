# Development Workflow

## GitHub Project Setup Guide

### 1. Create Repository on GitHub

#### Step 1: Initialize on GitHub
1. Go to github.com and sign in
2. Click "New" (top-left) or use github.com/new
3. Fill in repository details:
   - **Repository name**: `banking-fraud-detection`
   - **Description**: `Binary classification ML model for detecting fraudulent banking transactions`
   - **Visibility**: Public (for portfolio)
   - **Initialize with README**: No (we have one)
   - **Add .gitignore**: Python (optional, we have one)
   - **License**: MIT

4. Click "Create repository"

#### Step 2: Push Local Code
```bash
# Navigate to project directory
cd banking-fraud-detection

# Initialize git (if not already done)
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Project structure and documentation"

# Add remote origin
git remote add origin https://github.com/yourusername/banking-fraud-detection.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

---

## Branching Strategy

Use **Git Flow** model for organized development:

```
main (production-ready, tagged releases)
  ↑
release/v1.0.0 (release preparation)
  ↑
develop (integration branch)
  ↑
feature/* (feature development)
bugfix/* (bug fixes)
experiment/* (experimental work)
```

### Branch Naming Conventions

```
feature/feature-name          # New features
  feature/hyperparameter-tuning
  feature/ensemble-model
  
bugfix/bug-name              # Bug fixes
  bugfix/data-leakage-fix
  
docs/documentation-name      # Documentation
  docs/api-documentation
  
refactor/component-name      # Code refactoring
  refactor/preprocessing-module
```

### Creating a Feature Branch

```bash
# Update develop branch
git checkout develop
git pull origin develop

# Create feature branch
git checkout -b feature/new-feature-name

# Make changes and commit
git add .
git commit -m "Descriptive commit message"

# Push to remote
git push origin feature/new-feature-name

# Create Pull Request on GitHub
```

---

## Commit Message Standards

Write clear, descriptive commit messages following this format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types
- **feat**: A new feature
- **fix**: A bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting)
- **refactor**: Code refactoring
- **perf**: Performance improvements
- **test**: Adding or updating tests
- **chore**: Maintenance tasks

### Examples

```
feat(model): add ensemble voting classifier

- Implement voting classifier combining LR, RF, and XGB
- Soft voting with weighted estimators
- Achieves 0.918 ROC-AUC score

Closes #42
```

```
fix(preprocessing): resolve data leakage in feature engineering

- Features now computed only on training set
- Scaler fit on training data only
- Test set preprocessing uses fitted scaler

Fixes #31
```

```
docs(readme): update results section with final metrics

- Add model performance comparison table
- Include feature importance rankings
```

---

## Pull Request (PR) Process

### Creating a PR

1. **Push your branch to GitHub**
   ```bash
   git push origin feature/your-feature
   ```

2. **Go to GitHub repository**
   - Click "Pull requests" tab
   - Click "New pull request"
   - Select base: `develop`, compare: `feature/your-feature`

3. **Fill PR template**
   ```markdown
   ## Description
   Brief description of changes
   
   ## Type of change
   - [ ] New feature
   - [ ] Bug fix
   - [ ] Documentation update
   
   ## Related Issues
   Closes #123
   
   ## Testing
   Steps to test the changes
   
   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Tests written/updated
   - [ ] Documentation updated
   - [ ] No breaking changes
   ```

4. **Request reviewers** and wait for approval

### PR Review Checklist

Reviewers should check:
- ✅ Code quality and style
- ✅ Tests provided and passing
- ✅ No breaking changes
- ✅ Documentation updated
- ✅ Commit messages are clear

---

## Local Development Workflow

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/yourusername/banking-fraud-detection.git
cd banking-fraud-detection

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Install development dependencies
pip install pytest black flake8 jupyter
```

### Development Cycle

```bash
# 1. Create feature branch
git checkout -b feature/your-feature

# 2. Make changes
# Edit files, run notebooks, etc.

# 3. Test locally
pytest tests/

# 4. Format code
black src/

# 5. Lint
flake8 src/

# 6. Commit changes
git add .
git commit -m "feat(model): description of change"

# 7. Push to remote
git push origin feature/your-feature

# 8. Create PR on GitHub
# (See PR Process above)
```

---

## Code Quality Standards

### Style Guide
- Python: PEP 8 compliance
- Line length: 88 characters (Black default)
- Use descriptive variable/function names

### Code Quality Tools

#### Black (Code Formatter)
```bash
# Format all Python files
black src/

# Format specific file
black src/model_trainer.py
```

#### Flake8 (Linter)
```bash
# Check all Python files
flake8 src/

# Show statistics
flake8 src/ --statistics
```

#### Pytest (Testing)
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_preprocessing.py

# Run with coverage
pytest --cov=src tests/
```

---

## Project Structure Maintenance

### When Adding New Files

1. **Source Code** (`src/`)
   - Add implementation file
   - Add corresponding test file in `tests/`
   - Update docstrings

2. **Notebooks** (`notebooks/`)
   - Use sequential numbering: `01_`, `02_`, etc.
   - Add clear descriptions in notebook title
   - Include markdown cells explaining each section

3. **Documentation** (`docs/`)
   - Update relevant .md files
   - Keep README.md current
   - Add cross-references

4. **Data** (`data/`)
   - Raw data in `raw/` (never modify)
   - Processed data in `processed/`
   - Add data description files

---

## Continuous Integration (Optional)

### GitHub Actions Workflow

Create `.github/workflows/ci.yml`:

```yaml
name: CI Pipeline

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.8', '3.9', '3.10']
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov black flake8
    
    - name: Lint with flake8
      run: flake8 src/
    
    - name: Format check with black
      run: black --check src/
    
    - name: Test with pytest
      run: pytest tests/ --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

---

## Issue Tracking

### Creating Issues

Use GitHub Issues for:
- **Bug Reports**: Title starts with 🐛
- **Feature Requests**: Title starts with ✨
- **Documentation**: Title starts with 📚

### Issue Template

```markdown
## Description
Clear description of the issue

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- Python version: 3.9
- OS: macOS
- Relevant packages: scikit-learn 1.3.0, xgboost 2.0.0

## Screenshots
(if applicable)
```

---

## Release Management

### Creating a Release

1. **Update version numbers**
   - Update version in `config.py`
   - Update version in `setup.py` (if you have one)

2. **Create release branch**
   ```bash
   git checkout -b release/v1.1.0
   ```

3. **Update CHANGELOG**
   ```markdown
   ## [1.1.0] - 2026-05-20
   ### Added
   - New ensemble voting classifier
   - Feature importance visualization
   
   ### Fixed
   - Data leakage in preprocessing
   
   ### Changed
   - Improved model training speed
   ```

4. **Create PR to main**
   - Title: "Release v1.1.0"
   - Merge when approved

5. **Create GitHub Release**
   - Go to Releases page
   - Click "Create a new release"
   - Tag: `v1.1.0`
   - Title: `Version 1.1.0`
   - Description: Copy from CHANGELOG

---

## Documentation Standards

### Docstring Format (Google Style)

```python
def train_model(X_train, y_train, model_type='random_forest'):
    """
    Train a fraud detection model.
    
    Trains and evaluates a classification model on the provided training data.
    Supports multiple model types with automatic hyperparameter tuning.
    
    Args:
        X_train (pd.DataFrame): Features for training (n_samples, n_features)
        y_train (pd.Series): Binary fraud labels (0 or 1)
        model_type (str): Type of model. Options: 'random_forest', 'xgboost', 'logistic_regression'
    
    Returns:
        sklearn model: Trained and fitted model object
    
    Raises:
        ValueError: If model_type is not supported
        ValueError: If X_train and y_train have mismatched lengths
    
    Examples:
        >>> X_train = pd.DataFrame(...)
        >>> y_train = pd.Series(...)
        >>> model = train_model(X_train, y_train, model_type='xgboost')
        >>> predictions = model.predict(X_test)
    
    Note:
        Features should be already preprocessed and scaled if necessary.
        The model uses 5-fold cross-validation for evaluation.
    """
```

---

## Collaboration Best Practices

### Do's ✅
- Write clear commit messages
- Keep branches focused on single features
- Update documentation when changing code
- Test your code before pushing
- Review PRs thoroughly
- Use meaningful variable names
- Comment complex logic

### Don'ts ❌
- Commit large data files
- Commit credentials or API keys
- Push directly to main branch
- Ignore failing tests
- Write vague commit messages
- Leave commented-out code
- Make giant commits mixing multiple changes

---

## Useful Git Commands

```bash
# View commit history
git log --oneline

# View changes before committing
git diff

# Undo last commit (keep changes)
git reset --soft HEAD~1

# View remote branches
git branch -r

# Delete local branch
git branch -d feature/old-feature

# Sync with upstream
git fetch origin
git rebase origin/main

# View contributors
git log --format='%an' | sort | uniq -c | sort -rn
```

---

## Getting Help

### Resources
- GitHub Docs: https://docs.github.com
- Git Cheat Sheet: https://git-scm.com/docs
- Python Style Guide (PEP 8): https://pep8.org

### Questions?
- Check existing Issues/Discussions
- Create a new Discussion for questions
- Reference relevant documentation

---

**Last Updated**: May 2026
