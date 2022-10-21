from setuptools import setup

requirements = open('requirements.txt').readlines()
setup(
    name='TrafficGen',
    version='',
    packages=['TrafficGen_act', 'TrafficGen_act.data_process', 'TrafficGen_init', 'TrafficGen_init.data_process'],
    package_dir={'': 'trafficgen'},
    url='',
    license='',
    author='fenglan',
    author_email='',
    description='',
    install_requires = requirements
)
