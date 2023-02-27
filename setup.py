from setuptools import find_namespace_packages
from setuptools import setup

requirements = open('requirements.txt').readlines()

packages = find_namespace_packages()

packages = [f for f in packages if "trafficgen" in f]

setup(name='trafficgen', packages=packages, install_requires=requirements)
