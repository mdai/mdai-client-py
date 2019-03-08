from setuptools import setup
from setuptools import find_packages
from mdai import __version__

with open("README.md") as f:
    long_description = f.read()

setup(
    name="mdai",
    version=__version__,
    description="MD.ai Python client library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="MD.ai",
    author_email="github@md.ai",
    url="https://github.com/mdai/mdai-client-py",
    download_url="https://github.com/mdai/mdai-client-py/tarball/{}".format(__version__),
    license="Apache-2.0",
    install_requires=[
        "arrow",
        "matplotlib",
        "numpy",
        "opencv-python",
        "pandas",
        "pillow",
        "pydicom",
        "requests",
        "retrying",
        "scikit-image",
        "tqdm",
    ],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=find_packages(),
)
