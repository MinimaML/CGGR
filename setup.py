from setuptools import setup, find_packages
import pathlib
import sys

# Read the README for long description
here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

# Triton is only available on CUDA platforms (Linux/Windows)
# Make it optional for other platforms
install_requires = [
    "torch>=2.0.0",
]

setup(
    name="cggr",
    version="0.5.0",  # Bumped for architecture compatibility improvements
    description="Confidence-Gated Gradient Routing for Efficient Transformer Training",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MinimaML/CGGR",
    author="MinimaML",
    py_modules=["cggr", "triton_kernels", "cggr_async", "cggr_checkpointing", "cggr_dataloader", "cggr_flash"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
    ],
    install_requires=install_requires,
    extras_require={
        "cuda": ["triton>=2.0.0"],  # For Triton kernel acceleration
        "flash": ["flash-attn>=2.0.0"],  # For Flash Attention 2
        "dev": ["pytest", "rich"],
        "benchmark": ["transformers", "rich"],
        "all": ["triton>=2.0.0", "flash-attn>=2.0.0", "transformers", "rich", "pytest"],
    },
    python_requires=">=3.8",
)
