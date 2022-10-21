from setuptools import setup

requirements = open('requirements.txt').readlines()
setup(
    name='trafficgen',
    install_requires = requirements
)
