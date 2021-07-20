from pathlib import Path

from setuptools import find_packages, setup


def get_readme():
    """Load README.rst for display on PyPI."""
    with open("README.rst") as fhandle:
        return fhandle.read()


def get_extra_requires(path: str):
    req = Path(path).read_text()
    req = req.split('.. tab-end')[1]
    req = req.strip()
    req = req.split('\n')

    req_dict = {'all': set()}
    for r in req:
        pack, key = (_.strip() for _ in r.split(':'))
        if not any(special in pack for special in ('scm', 'optsam')):  # Cannot be requested from PyPI.
            req_dict['all'].add(pack)

        if key in req_dict:
            req_dict[key].add(pack)
        else:
            req_dict[key] = {pack}

    return req_dict


with open('glompo/_version.py', 'r') as file:
    exec(file.read())

setup(
    name="glompo",
    version=__version__,
    description="Globally managed parallel optimization",
    long_description=get_readme(),
    author="Michael Freitas Gustavo",
    author_email="michael.freitasgustavo@ugent.be",
    url="https://github.com/mfgustavo/glompo",
    download_url="https://github.com/mfgustavo/glompo",
    packages=find_packages(),
    license_file='LICENSE',
    include_package_data=True,
    package_dir={"glompo": "glompo"},
    install_requires=['numpy~=1.17.4', 'PyYAML~=5.1.2', 'tables~=3.6.1'],
    python_requires='>=3.6',
    extras_require=get_extra_requires('extra_requirements.txt'),
)
