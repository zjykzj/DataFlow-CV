#!/usr/bin/env python3
"""Setup script for DataFlow-CV."""

from setuptools import setup, find_packages
import re

# Read version from dataflow/__init__.py
with open('dataflow/__init__.py', 'r') as f:
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    version = version_match.group(1) if version_match else "0.1.0"

# Read long description from README.md
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# Runtime dependencies
install_requires = [
    'numpy>=1.24.0',
    'opencv-python>=4.6.0.66',
    'click>=7.0.0',
]

# Optional dependencies (extras)
extras_require = {
    'coco': ['pycocotools>=2.0.0'],
    'dev': [
        'pytest>=7.0.0',
        'pytest-cov>=4.0.0',
        'black>=22.0.0',
        'isort>=5.12.0',
        'flake8>=6.0.0',
        'mypy>=1.0.0',
        'pylint>=3.0.0',
    ],
    'docs': [
        'sphinx>=7.0.0',
        'sphinx-rtd-theme>=1.3.0',
    ],
}

setup(
    name='dataflow-cv',
    version=version,
    author='DataFlow-CV Team',
    author_email='example@example.com',
    description='A computer vision dataset processing library',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/DataFlow-CV',
    packages=find_packages(include=['dataflow', 'dataflow.*']),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='>=3.8',
    install_requires=install_requires,
    extras_require=extras_require,
    include_package_data=True,
    zip_safe=False,
    keywords='computer-vision, dataset, annotation, label, labelme, yolo, coco',
    project_urls={
        'Bug Reports': 'https://github.com/yourusername/DataFlow-CV/issues',
        'Source': 'https://github.com/yourusername/DataFlow-CV',
    },
)