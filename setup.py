from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open("requirements.txt") as req_file:
    requirements = list(filter(None, req_file.read().split("\n")))

__version__ = None
with open("sleeputils/version.py") as version_file:
    exec(version_file.read())
if __version__ is None:
    raise ValueError("Did not find __version__ in version.py file.")

setup(
    name='sleeputils',
    version=__version__,
    description='A collection of Python utility classes and functions for loading and processing Polysomnography (PSG) sleep study data.',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='Mathias Perslev',
    author_email='map@di.ku.dk',
    url='https://github.com/perslev/U-Sleep-API-Python-Bindings',
    packages=find_packages(),
    package_dir={'sleeputils':
                 'sleeputils'},
    include_package_data=True,
    install_requires=requirements,
    classifiers=['Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7',
                 'Programming Language :: Python :: 3.8',
                 'Programming Language :: Python :: 3.9',
                 'Programming Language :: Python :: 3.10',
                 'Programming Language :: Python :: 3.11']
)
