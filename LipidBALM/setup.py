from setuptools import setup, find_packages

setup(
    name="balm",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'torch',
        'transformers',
        'pandas',
        # Add other dependencies
    ],
)

