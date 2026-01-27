"""Setup configuration for MLOps project."""
from setuptools import setup, find_packages

setup(
    name="mlops-pipeline",
    version="0.1.0",
    description="MLOps pipeline for scalable ML model management",
    author="MLOps Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "mlflow>=2.0.0",
        "pyyaml>=6.0",
        "joblib>=1.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
)
