from setuptools import setup, find_packages

setup(
    name="psi_calculator",
    version="0.1.0",
    author="Your Name",
    author_email="your_email@example.com",
    description="A package for Population Stability Index (PSI) calculation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
