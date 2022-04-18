from setuptools import find_packages, setup

__version__ = '2.0.5'
URL = 'https://github.com/pyg-team/pytorch_geometric'

install_requires = [
    'tqdm',
    'numpy',
    'scipy',
    'jinja2',
    'requests',
    'pyparsing',
    'scikit-learn',
]

full_install_requires = [
    'h5py',
    'yacs',
    'numba',
    'pandas',
    'captum',
    'rdflib',
    'trimesh',
    'networkx',
    'tabulate',
    'hydra-core',
    'matplotlib',
    'scikit-image',
    'pytorch-memlab',
    'torchmetrics>=0.7',
]

test_requires = [
    'pytest',
    'pytest-cov',
]

dev_requires = test_requires + [
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
        'deep-learning',
        'pytorch',
        'geometric-deep-learning',
        'graph-neural-networks',
        'graph-convolutional-networks',
    ],
    python_requires='>=3.7',
    install_requires=install_requires,
    extras_require={
        'full': full_install_requires,
        'test': test_requires,
        'dev': dev_requires,
    },
    packages=find_packages(),
    include_package_data=True,
)
