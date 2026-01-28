"""
Setup configuration for PINECONe package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="pinecone-fire",
    version="0.1.0",
    author="NASA EarthRISE",
    author_email="contact@earthrise.nasa.gov",  # Update with actual email
    description="Pine Ecosystem Carbon and Economics RX Fire Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NASA-EarthRISE/PINECONe",
    project_urls={
        "Bug Tracker": "https://github.com/NASA-EarthRISE/PINECONe/issues",
        "Documentation": "https://nasa-earthrise.github.io/PINECONe/",
        "Source Code": "https://github.com/NASA-EarthRISE/PINECONe",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "earthengine-api>=0.1.380",
        "geemap>=0.30.0",
        "geopandas>=0.14.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "rasterio>=1.3.0",
        "shapely>=2.0.0",
        "pyyaml>=6.0",
        "seaborn>=0.12.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "sphinx>=7.0.0",
            "jupyter>=1.0.0",
            "notebook>=7.0.0",
        ],
    },
    keywords=[
        "fire",
        "carbon",
        "remote-sensing",
        "google-earth-engine",
        "forestry",
        "prescribed-fire",
        "biomass",
        "economics",
        "longleaf-pine",
    ],
    include_package_data=True,
)
