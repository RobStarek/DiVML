from setuptools import setup, find_packages

setup(
    name='dvml',
    version='0.1.0',
    description='Discrete-variable quantum maximum-likelihood reconstruction.',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch',
        'numba',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)