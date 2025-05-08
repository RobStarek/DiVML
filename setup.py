from setuptools import setup, find_packages

setup(
    name='DiVML',
    version='0.1.0',
    description='Discrete-variable quantum state maximum-likelihood reconstruction.',
    author='Robert Starek',
    author_email='starek.robert@gmail.com',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch',
        'numba',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',        
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)