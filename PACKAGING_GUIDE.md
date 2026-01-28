# Making PINECONe Pip-Installable

This guide walks you through making PINECONe installable via pip.

## 📋 Files Created

1. **setup.py** - Package setup script (backward compatible)
2. **pyproject.toml** - Modern Python packaging configuration (PEP 517/518)
3. **MANIFEST.in** - Specifies which files to include in distribution
4. **requirements.txt** - List of dependencies

---

## 🚀 Quick Start - Install from GitHub

Users can now install directly from GitHub:

```bash
pip install git+https://github.com/NASA-EarthRISE/PINECONe.git
```

---

## 🛠️ Local Development Installation

### Step 1: Add Files to Repository

```powershell
cd C:\Users\Mayer\Documents\GitHub\PINECONe

# Add packaging files
git add setup.py
git add pyproject.toml
git add MANIFEST.in
git add requirements.txt

# Commit
git commit -m "Add packaging configuration for pip installation"

# Push
git push origin main
```

### Step 2: Install in Development Mode

```bash
# From your local repository
cd C:\Users\Mayer\Documents\GitHub\PINECONe

# Install in editable mode (changes reflect immediately)
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
```

### Step 3: Test the Installation

```python
# Test import
python -c "from pinecone.data.biomass import BiomassData; print('✓ Package installed successfully!')"
```

---

## 📦 Building Distribution Packages

To create installable packages (.tar.gz and .whl):

```bash
# Install build tools
pip install build

# Build packages
python -m build

# This creates:
# dist/pinecone-fire-0.1.0.tar.gz
# dist/pinecone_fire-0.1.0-py3-none-any.whl
```

---

## 🌐 Publishing to PyPI

### Option 1: TestPyPI (Recommended First)

Test your package before publishing to real PyPI:

```bash
# 1. Create account at https://test.pypi.org/account/register/

# 2. Install twine
pip install twine

# 3. Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# 4. Test installation
pip install --index-url https://test.pypi.org/simple/ pinecone-fire
```

### Option 2: Real PyPI

Once tested on TestPyPI:

```bash
# 1. Create account at https://pypi.org/account/register/

# 2. Upload to PyPI
python -m twine upload dist/*

# 3. Install from PyPI
pip install pinecone-fire
```

---

## 🔐 PyPI Authentication

### Using API Tokens (Recommended)

1. Go to https://pypi.org/manage/account/token/
2. Create a new API token
3. Create `~/.pypirc`:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-your-api-token-here

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-your-test-token-here
```

Then upload without entering credentials:
```bash
python -m twine upload dist/*
```

---

## 📝 Version Management

### Updating Version Number

Edit **both** files when releasing new versions:

1. **setup.py**: Change `version="0.1.0"` to `version="0.1.1"`
2. **pyproject.toml**: Change `version = "0.1.0"` to `version = "0.1.1"`

### Semantic Versioning

- **0.1.0** → **0.1.1**: Bug fixes
- **0.1.0** → **0.2.0**: New features (backward compatible)
- **0.1.0** → **1.0.0**: Breaking changes

---

## ✅ Pre-Publishing Checklist

- [ ] All tests pass: `pytest tests/`
- [ ] Documentation is up to date
- [ ] README.md is comprehensive
- [ ] LICENSE file exists
- [ ] Version number is updated
- [ ] CHANGELOG.md is updated (optional but recommended)
- [ ] Tested local installation: `pip install -e .`
- [ ] Tested GitHub installation: `pip install git+https://github.com/...`
- [ ] Built distribution: `python -m build`
- [ ] Uploaded to TestPyPI first
- [ ] Tested TestPyPI installation

---

## 🎯 Current Status

After adding these files and pushing to GitHub, users can install with:

```bash
# Install from GitHub (works immediately)
pip install git+https://github.com/NASA-EarthRISE/PINECONe.git

# Install from specific branch
pip install git+https://github.com/NASA-EarthRISE/PINECONe.git@main

# Install from specific commit
pip install git+https://github.com/NASA-EarthRISE/PINECONe.git@abc123

# Install in editable mode for development
git clone https://github.com/NASA-EarthRISE/PINECONe.git
cd PINECONe
pip install -e .
```

---

## 🔧 Common Issues

### Issue: "Package not found"
**Solution**: Make sure `src/pinecone/__init__.py` exists

### Issue: "Module not found" when importing
**Solution**: Install in editable mode: `pip install -e .`

### Issue: Dependencies not installing
**Solution**: Install them manually: `pip install -r requirements.txt`

### Issue: "No module named 'setuptools'"
**Solution**: `pip install --upgrade setuptools wheel`

---

## 📚 Next Steps

1. **Test locally**: `pip install -e .`
2. **Update Colab notebook**: Change install command to GitHub URL
3. **Test from GitHub**: `pip install git+https://github.com/NASA-EarthRISE/PINECONe.git`
4. **Publish to TestPyPI**: Test the full PyPI workflow
5. **Publish to PyPI**: Make it publicly pip-installable

---

## 🎉 Success!

Once these files are committed and pushed, anyone can install PINECONe with:

```bash
pip install git+https://github.com/NASA-EarthRISE/PINECONe.git
```

And after publishing to PyPI:

```bash
pip install pinecone-fire
```
