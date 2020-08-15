
from setuptools import setup


def readme():
    with open('README.md') as f:
        README = f.read()
    return README




setup(
    name="py-automl",
    version="1.0.0",
    description="py-automl - An open source, low-code machine learning library in Python.",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/PrudhviGNV/py-automl",
    author="Prudhvi GNV",
    author_email="prudhvi.gnv@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    packages=["py-automl"],
    include_package_data=True,
    install_requires=required
)
