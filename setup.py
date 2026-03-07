from setuptools import setup, find_packages

setup(
    name="dataflow-cv",
    version="0.1.1",
    description="A data processing library for computer vision datasets",
    author="DataFlow Team",
    license="MIT",
    packages=find_packages(include=["dataflow*"]),
    install_requires=[
        "numpy>=2.0.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "click>=8.1.0",
    ],
    extras_require={
        "full": [
            "pycocotools>=2.0.0",
            "torch>=1.9.0",
            "torchvision>=0.10.0",
        ]
    },
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "dataflow=dataflow.cli:main"
        ]
    },
)