from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="llminfer",
    version="0.1.0",
    author="LogicIF Team",
    description="A simple package for LLM inference with JSONL file support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Core dependencies - users install provider packages as needed
    ],
    extras_require={
        "openai": ["openai>=1.0.0"],
        "anthropic": ["anthropic>=0.8.0"],
        "gemini": ["google-generativeai>=0.3.0"],
        "vllm": ["vllm>=0.2.0"],
        "all": [
            "openai>=1.0.0",
            "anthropic>=0.8.0", 
            "google-generativeai>=0.3.0",
            "vllm>=0.2.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "llminfer=llminfer.cli:main",
        ],
    },
) 