"""Setup script for the sentisynth package."""
from setuptools import setup, find_packages

setup(
    name="sentisynth",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.10.0",
        "datasets>=1.11.0",
        "numpy>=1.19.5",
        "scikit-learn>=0.24.2",
        "pandas>=1.3.0",
        "matplotlib>=3.4.2",
        "seaborn>=0.11.1",
        "tqdm>=4.61.2",
    ],
    entry_points={
        "console_scripts": [
            "sentisynth=sentisynth.cli:main",
        ],
    },
    author="Param Kapur",
    author_email="kpr.param@gmail.com",
    description="Synthetic data generation for sentiment analysis",
    keywords="nlp, sentiment-analysis, synthetic-data",
    url="https://github.com/paramkpr/sentisynth",
)
