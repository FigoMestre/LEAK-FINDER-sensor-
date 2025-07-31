#!/usr/bin/env python3
"""
Setup script for LEAK-FINDER Sensor
Câmera Acústica de Baixo Custo
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), '..', 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "LEAK-FINDER Sensor - Câmera Acústica de Baixo Custo"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="leak-finder-sensor",
    version="1.0.0",
    author="LEAK-FINDER Team",
    author_email="contact@leak-finder.com",
    description="Câmera Acústica de Baixo Custo para Detecção de Vazamentos",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/seu-usuario/LEAK-FINDER-sensor-",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Manufacturing",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
    ],
    python_requires=">=3.10",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
        ],
        "optional": [
            "acoular>=22.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "leak-finder=acoustic_camera:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
) 