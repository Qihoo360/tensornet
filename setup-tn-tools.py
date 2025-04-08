from setuptools import setup
from importlib.machinery import SourceFileLoader

# use importlib to avoid import so file
_version = SourceFileLoader("version", "tensornet/version.py").load_module()
version = _version.VERSION


setup(
    name="qihoo-tensornet-tools",
    version=version,
    description="tools for tensornet",
    long_description="multi tools for tensornet. E.g. merge/resize sparse or dense table, include external embeddings",
    long_description_content_type="text/markdown",
    author="jiangxinglei",
    author_email="jiangxinglei@360.cn",
    url="https://github.com/Qihoo360/tensornet",
    packages=["tensornet-tools"],
    package_data={"tensornet-tools": ["bin/*", "config/*", "python/*"]},
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.7",
    ],
)
