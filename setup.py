from setuptools import setup, Extension, find_packages
import io
import codecs
import os
import sys

import orcs
import orcs.version

packages = find_packages(where=".")

here = os.path.abspath(os.path.dirname(__file__))

with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
    


setup(
    name='orcs',
    version=orcs.version.__version__,
    url='https://github.com/thomasorb/orcs',
    license='GPLv3+',
    author='Thomas Martin',
    author_email='thomas.martin.1@ulaval.ca',
    maintainer='Thomas Martin',
    maintainer_email='thomas.martin.1@ulaval.ca',
    description='Analysis engine for SITELLE spectral cubes',
    long_description=long_description,
    packages=packages,
    package_dir={"": "."},
    include_package_data=True,
    package_data={
        '':['LICENSE.txt', '*.rst', '*.txt', 'docs/*', '*.pyx'],
        'orcs':['data/*', '*.pyx']},
    exclude_package_data={
        '': ['*~', '*.so', '*.pyc'],
        'orcs':['*~', '*.so', '*.pyc', '*.c']},
    platforms='any',
    classifiers = [
        'Programming Language :: Python',
        'Programming Language :: Cython',
        'Development Status :: 4 - Beta',
        'Natural Language :: English',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: OS Independent' ],
)
