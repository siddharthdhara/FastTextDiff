from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="fasttextdiff",
    version="0.1.0",
    author="Venkata Siddharth Dhara, Pawan Kumar",
    author_email="",  # Add your email
    description="A Fast and Efficient Modern BERT based Text-Conditioned Diffusion Model for Medical Image Segmentation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/siddharthdhara/FastTextDiff",
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
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
            "mypy",
        ],
    },
    entry_points={
        "console_scripts": [
            "fasttextdiff-train=src.training.train:main",
            "fasttextdiff-eval=src.evaluation.eval:main",
        ],
    },
)
