

from setuptools import setup, find_packages


def get_readme():
    """Load README.rst for display on PyPI."""
    with open("README.rst") as fhandle:
        return fhandle.read()


setup(
    name="glompo",
    version="1.0.0",
    description="Globally managed parallel optimization",
    long_description=get_readme(),
    author="Michael Freitas Gustavo",
    author_email="michael.freitasgustavo@ugent.be",
    url="https://github.com/mfgustavo/glompo",
    packages=find_packages(),
    include_package_data=True,
    package_dir={"glompo": "glompo"},
    install_requires=['numpy', 'matplotlib', 'scipy', 'pytest', 'cma', 'PyYAML', 'emcee', 'nevergrad', 'scm']
)
