from setuptools import setup, find_packages

__version__ = '2.0.4'
URL = 'https://github.com/pyg-team/pytorch_geometric'

install_requires = [
    'tqdm',
    'yacs',
    'numpy',
    'scipy',
    'pandas',
    'jinja2',
    'pyparsing',
    'scikit-learn',
    'googledrivedownloader',
]

full_install_requires = [
    'h5py',
    'numba',
    'rdflib',
    'trimesh',
    'networkx',
    'tabulate',
    'matplotlib',
    'scikit-image',
    'pytorch-memlab',
]

tests_require = [
    'pytest',
    'pytest-cov',
]

dev_requires = tests_require + [
    'pre-commit',
]

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
    extras_require={
        'full': full_install_requires,
        'test': tests_require,
        'dev': dev_requires,
    },
    packages=find_packages(),
    include_package_data=True,
)
