import os
from setuptools import setup, find_packages

version = "0.13.0.dev0"

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
    install_requires=[
        'tensorflow>=2.2,<2.3'
    ],
    python_requires='>=3.7, <3.8',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.7'
    ],
)
