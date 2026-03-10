from setuptools import setup, find_packages
from setuptools.command.develop import develop as _develop
import os
import sys

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


class DevelopCommand(_develop):
    """Custom develop command that doesn't call pip."""

    def run(self):
        # Don't call parent's run() because it calls pip
        # Instead, just set up the egg-link

        # Initialize options
        self.initialize_options()
        self.finalize_options()

        # Create egg-info directory
        self.run_command("egg_info")

        # Create egg-link
        egg_link = os.path.join(self.install_dir, f"{self.distribution.get_name()}.egg-link")
        os.makedirs(self.install_dir, exist_ok=True)

        with open(egg_link, 'w') as f:
            f.write(os.path.abspath(self.egg_base) + '\n')
            # Write empty second line (traditional egg-link format)
            f.write('.')

        print(f"Created egg-link: {egg_link}")
        print(f"Editable installation complete. Use 'python -m dataflow.cli' to run CLI.")


setup(
    name="dataflow-cv",
    version="0.3.0",
    author="DataFlow Team",
    description="A data processing library for computer vision datasets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zjykzj/DataFlow-CV",
    license="MIT",
    packages=find_packages(include=["dataflow*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=2.0.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "click>=8.1.0",
    ],
    entry_points={
        "console_scripts": [
            "dataflow=dataflow.cli:main"
        ]
    },
    cmdclass={
        "develop": DevelopCommand,
    },
)