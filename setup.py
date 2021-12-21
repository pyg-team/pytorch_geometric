from setuptools import setup, find_packages

__version__ = '2.0.3'
URL = 'https://github.com/pyg-team/pytorch_geometric'

install_requires = [
    'numpy',
    'tqdm',
    'scipy',
    'networkx',
    'scikit-learn',
    'requests',
    'pandas',
    'rdflib',
    'googledrivedownloader',
    'jinja2',
    'pyparsing',
    'yacs',
    'PyYAML',
]
tests_require = ['pytest', 'pytest-cov', 'mock']

setup(
    name='torch_geometric',
    version=__version__,
    description='Graph Neural Network Library for PyTorch',
    author='Matthias Fey',
    author_email='matthias.fey@tu-dortmund.de',
    url=URL,
    download_url=f'{URL}/archive/{__version__}.tar.gz',
    keywords=[
        'deep-learning'
        'pytorch',
        'geometric-deep-learning',
        'graph-neural-networks',
        'graph-convolutional-networks',
    ],
    python_requires='>=3.6',
    install_requires=install_requires,
    extras_require={'test': tests_require},
    packages=find_packages(),
    include_package_data=True,
)
