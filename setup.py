from setuptools import setup, find_packages

setup(
    name="compactpy",
    version="0.1.0",
    packages=find_packages(include=["compactpy", "compactpy.*"]),
    install_requires=[
        "tiktoken>=0.5.0",
    ],
    description="Intelligent Context Compression Framework",
    author="Priyankar Majumdar",
    python_requires=">=3.9",
)