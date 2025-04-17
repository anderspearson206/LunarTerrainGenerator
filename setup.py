from setuptools import setup, find_packages

setup(
    name='lunar_terrain_generator',
    version='0.1.0',
    author='Anders Pearson',
    author_email='amp206@uw.edu',
    description='A simple lunar terrain generator',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/anderspearson206/LunarTerrainGenerator',
    packages=find_packages(),
    install_requires=open('requirements.txt').read().splitlines(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)