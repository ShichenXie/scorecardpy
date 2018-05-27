# -*- coding: utf-8 -*-

"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject

https://python-packaging.readthedocs.io/
https://packaging.python.org/tutorials/distributing-packages/
"""

# Always prefer setuptools over distutils
from setuptools import setup#, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))
# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='scorecardpy',  # Required
    version='0.1.4',  # Required
    description='Credit Risk Scorecard',  # Required
    long_description=long_description,  # Optional
    long_description_content_type='text/markdown',  # Optional (see note above)
    url='http://github.com/shichenxie/scorecardpy',  # Optional
    author='Shichen Xie',  # Optional
    author_email='xie@shichen.name',  # Optional
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        # 'Topic :: Software Development :: Build Tools',

        # Pick your license as you wish
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='credit scorecard, woe binning, performace evaluation',  # Optional
    packages=['scorecardpy'],  # Required
    install_requires=['numpy','pandas>=0.22.0','matplotlib','scikit-learn>=0.19.1'],  # Optional
    package_data={'scorecardpy': ['data/*.csv']},
    # data_files=[('scorecardpy': ['data/*.csv'])],  # Optional
    project_urls={  # Optional
        'Bug Reports': 'https://github.com/shichenxie/scorecardpy/issues',
        'Source': 'https://github.com/shichenxie/scorecardpy/',
    },
)
