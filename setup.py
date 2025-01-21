from setuptools import setup, find_packages

setup(
    name="next-random-number-identifier",
    version="2.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "matplotlib"
    ],
    python_requires=">=3.8",
    author="AIQube Centaur Systems Team",
    description="A machine learning-based system for analyzing and forecasting random number sequences",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
)