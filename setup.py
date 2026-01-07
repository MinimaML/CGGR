from setuptools import setup, find_packages
import pathlib

# Read the README for long description
here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="cggr",
    version="0.1.0",
    description="Confidence-Gated Gradient Routing for Efficient Transformer Training",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MinimaML/CGGR",
    author="MinimaML",
    py_modules=["cggr", "triton_kernels"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    install_requires=[
        "torch>=2.0.0",
        "numpy",
    ],
    extras_require={
        "triton": ["triton>=2.0.0"],
    },
    python_requires=">=3.8",
)
