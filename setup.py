"""
PKS-MPNN: Fine-tuning ProteinMPNN for Polyketide Synthases

This package provides tools for:
- Processing PKS module structures from AlphaFold3 predictions
- Training ProteinMPNN variants on PKS-specific datasets
- Evaluating sequence design performance on PKS domains
"""

from setuptools import setup, find_packages

setup(
    name="pks-mpnn",
    version="0.1.0",
    author="PKS-MPNN Team",
    description="Fine-tuning ProteinMPNN for Polyketide Synthase sequence design",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/PKS-MPNN",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "numpy>=1.24",
        "pandas>=2.0",
        "scipy>=1.10",
        "biopython>=1.81",
        "pyyaml>=6.0",
        "tqdm>=4.65",
        "wandb>=0.15",
        "einops>=0.6",
        "omegaconf>=2.3",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=23.0",
            "isort>=5.12",
            "flake8>=6.0",
        ],
        "viz": [
            "matplotlib>=3.7",
            "seaborn>=0.12",
            "jupyter>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pks-prepare=scripts.prepare_data:main",
            "pks-train=scripts.train:main",
            "pks-evaluate=scripts.evaluate:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
