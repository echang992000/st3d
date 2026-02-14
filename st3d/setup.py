from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="st3d",
    version="0.1.0",
    description="3D spatial transcriptomics reconstruction from serial 2D sections",
    author="Eric",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        "dev": ["pytest", "black", "isort", "flake8"],
    },
)
