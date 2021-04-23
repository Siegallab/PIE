"""Required packages for install and tests."""

import os
import re

from setuptools import setup, find_packages


install_requires = [
    'numpy>=1.16.0',
    'opencv-contrib-python>4.0.0.0',
    'pandas>=1.1.0',
    'Pillow>=7.2.0',
    'scipy>=1.1.0',
    'plotnine >= 0.7.1',
    'pyarrow',
    'Click>=7.0',
    'scikit-learn>=0.23.0'
]


tests_require = [
    'nose'
]


extras_require = {
    'tests': tests_require
}


# Get the version string from the version.py file.
# Based on:
# https://stackoverflow.com/questions/458550
with open(os.path.join('PIE', 'version.py'), 'rt') as f:
    filecontent = f.read()
match = re.search(r"^__version__\s*=\s*['\"]([^'\"]*)['\"]", filecontent, re.M)
if match is not None:
    version = match.group(1)
else:
    raise RuntimeError('unable to find version string in %s.' % (filecontent,))


# Get long project description text from the README.md file
with open('readme.md', 'rt') as f:
    readme = f.read()


setup(
    name='PIE',
    version=version,
    description='Library for ...',
    long_description=readme,
    long_description_content_type='text/markdown',
    keywords='...',
    url='https://github.com/Siegallab/PIE',
    author='Yevgeniy Plavskin, Shuang Li',
    author_email='eugene.plavskin@nyu.edu, sl2803@nyu.edu',
    license='New BSD',
    license_file='LICENSE',
    packages=find_packages(exclude=('PIE_tests',)),
    include_package_data=True,
    test_suite='nose.collector',
    extras_require=extras_require,
    tests_require=tests_require,
    install_requires=install_requires,
    entry_points={
        'console_scripts': [
            'pie = PIE.command_line_interface:cli',
        ]
    },
    classifiers=[
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python'
    ]
)
