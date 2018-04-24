# -*- coding: utf-8 -*-
from setuptools import setup


setup(name='scorecardpy',
      version='0.1.0',
      description='Credit Risk Scorecard',
      long_description='Makes the development of credit risk scorecard \
      easily and efficiently by providing functions such as information \
      value, variable filter, optimal woe binning, scorecard scaling and \
      performance evaluation etc.',
      keywords='scorecard, woebinning, performace evaluation',
      url='http://github.com/shichenxie/scorecardpy',
      author='Shichen Xie',
      author_email='xie@shichen.name',
      license='MIT',
      packages=['scorecardpy'],
      package_data={'scorecardpy': ['data/*.csv']},
      install_requires=[
          'numpy','pandas','matplotlib',
      ],
      include_package_data=True,
      zip_safe=False)


# https://python-packaging.readthedocs.io/
