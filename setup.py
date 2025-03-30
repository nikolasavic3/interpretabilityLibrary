from setuptools import setup, find_packages

setup(
    name="keras-interpret",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "keras>=3.0.0",
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
    ],
    
    description="Interpretability library for Keras",
    
)