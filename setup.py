import os
import re
from setuptools import setup, find_packages

base_package = 'skued'

# Get the version (borrowed from SQLAlchemy)
base_path = os.path.dirname(__file__)
with open(os.path.join(base_path, 'skued', '__init__.py')) as f:
    module_content = f.read()
    VERSION = re.compile(r'.*__version__ = \'(.*?)\'', re.S).match(module_content).group(1)
    LICENSE = re.compile(r'.*__license__ = \'(.*?)\'', re.S).match(module_content).group(1)


with open('README.rst') as f:
    readme = f.read()

with open('CHANGELOG.rst') as f:
    changes = f.read()

with open('requirements.txt') as f:
    requirements = [line for line in f.read().split('\n') if len(line.strip())]


packages = [base_package + '.' + x for x in find_packages(os.path.join(base_path, base_package))]
if base_package not in packages:
    packages.append(base_package)


if __name__ == '__main__':
    setup(
        name='skued',
        description='Collection of algorithms and functions for ultrafast electron diffraction',
        long_description='\n\n'.join([readme, changes]),
        license=LICENSE,
        url='http://skued.readthedocs.io',
        version=VERSION,
        author='Laurent P. René de Cotret',
        author_email='laurent.renedecotret@mail.mcgill.ca',
        maintainer='Laurent P. René de Cotret',
        maintainer_email='laurent.renedecotret@mail.mcgill.ca',
        entry_points={
            'console_scripts': [
                'skued = skued.cli:main'
            ]
        },
        install_requires=requirements,
        keywords=['skued'],
        packages=packages,
        zip_safe=False,
        classifiers=['Intended Audience :: Developers',
                     'License :: OSI Approved :: MIT License',
                     'Natural Language :: English',
                     'Operating System :: OS Independent',
                     'Programming Language :: Python :: 2',
                     'Programming Language :: Python :: 2.7',
                     'Programming Language :: Python :: 3',
                     'Programming Language :: Python :: 3.3',
                     'Programming Language :: Python :: 3.4',
                     'Programming Language :: Python :: 3.5',
                     'Programming Language :: Python :: 3.6']
    )
