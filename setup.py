import os
from setuptools import setup, find_packages
from importlib.machinery import SourceFileLoader

# use importlib to avoid import so file
_version = SourceFileLoader('version', 'tensornet/version.py').load_module()
version = _version.VERSION


setup(
    name='qihoo-tensornet',
    version=version,
    description='tensornet',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='jiangxinglei',
    author_email='jiangxinglei@360.cn',
    url='https://github.com/Qihoo360/tensornet',
    packages=find_packages(),
    package_data = {
        "tensornet.core": ["_pywrap_tn.so"],
    },
    python_requires='>=3.7, <3.8',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.7'
    ],
    platforms = ["manylinux2010_x86_64"],
)
