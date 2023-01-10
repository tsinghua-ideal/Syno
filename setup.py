from setuptools import setup

setup(
    name="KAS",
    version="0.0.1",
    description="A loop-level kernel architechure search tool",
    cmake_source_dir=".",
    packages=["kas"],
    zip_safe=False,
    install_requires=['torch>=1.10.0'],
    extras_require={"test": ["pytest>=6.0"]},
    python_requires=">=3.7",
)
