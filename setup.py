#!/usr/bin/env python
#
import os
from glob import glob
from setuptools import setup, find_packages
#
# Begin setup
#
setup_keywords = dict()
#
setup_keywords['name'] = 'snxform'
setup_keywords['description'] = 'A Python package for identifying supernovae in optical spectra'
setup_keywords['license'] = 'BSD'
setup_keywords['version'] = '1.0.0'
#
# Use README.md as a long description.
#
setup_keywords['long_description'] = ''
if os.path.exists('README.md'):
    with open('README.md') as readme:
        setup_keywords['long_description'] = readme.read()
    setup_keywords['long_description_content_type'] = 'text/markdown'
#
# Other keywords for the setup function.
#
setup_keywords['provides'] = [setup_keywords['name']]
setup_keywords['python_requires'] = '>=3.8'
setup_keywords['zip_safe'] = False
setup_keywords['packages'] = find_packages('python')
setup_keywords['package_dir'] = {'' : 'python'}
#
requires = []
with open('requirements.txt', 'r') as f:
    for line in f:
        if line.strip():
            requires.append(line.strip())
setup_keywords['install_requires'] = requires
#
# Internal data directories
#
setup_keywords['package_data'] = {'snxform': ['etc/*.ecsv', 'etc/*.pt', 'etc/*.json']}
setup_keywords['include_package_data'] = True
#
# Run setup command.
#
setup(**setup_keywords)
