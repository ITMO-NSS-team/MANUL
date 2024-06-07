from pathlib import Path
from typing import List

import setuptools

LOCATION = Path(__file__).parent.resolve()
NAME = 'MANUL'
VERSION = 1
README = Path(LOCATION, 'README.rst').read_text(encoding='utf-8')
REQ_PYTHON = '>=3.8'

def _readlines(*names: str, **kwargs) -> List[str]:
    encoding = kwargs.get('encoding', 'utf-8')
    lines = Path(__file__).parent.joinpath(*names).read_text(encoding=encoding).splitlines()
    return list(map(str.strip, lines))


def _extract_requirements(file_name: str):
    return [line for line in _readlines(file_name) if line and not line.startswith('#')]


def _get_requirements(req_name: str):
    requirements = _extract_requirements(req_name)
    return requirements

setuptools.setup(
    name=NAME,
    version=VERSION,
    long_description=README,
    long_description_content_type='text/x-rst',
    python_requires=REQ_PYTHON,
    extras_require={
        key: _get_requirements(Path('other_requirements', f'{key}.txt'))
        for key in ('evo_test')
    },
)