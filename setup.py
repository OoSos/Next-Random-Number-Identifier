from setuptools import setup, find_packages

setup(
    name="next-random-number-identifier",
    version="0.0.0",  # Version will be updated by semantic-release
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    description="A machine learning-based system for analyzing and forecasting random number sequences",
    author="OoSos",
    author_email="example@example.com",
    url="https://github.com/OoSos/next-random-number-identifier",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "matplotlib",
    ],
    extras_require={
        "dev": ["black", "isort", "flake8", "mypy"],
        "test": ["pytest", "pytest-cov"],
        "doc": ["sphinx", "sphinx_rtd_theme"],
    },
)