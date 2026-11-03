#!/usr/bin/env python

from distutils.core import setup

setup(name='cenrl',
      version='1.0a1',
      description='CenRL',
      author='Anonynmous',
      author_email='',
      url='',
      packages=['common', 'baselines', 'models', 'api', 'Infrastructure'],
      package_data={'models': ['inputs/*.csv']},
      install_requires=[
         'numpy',
         'pandas',
         'tqdm',
         'networkx',
         'braveblock',
         'matplotlib',
         'seaborn'
      ]
     )
