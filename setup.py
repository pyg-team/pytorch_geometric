from setuptools import find_packages, setup

__version__ = '2.1.0'
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

graphgym_requires = [
    'yacs',
    'hydra-core',
    'protobuf<4.21',
    'pytorch-lightning',
]

full_requires = graphgym_requires + [
    'ase',
    'h5py',
    'numba',
    'sympy',
    'pandas',
    'captum',
    'rdflib',
    'trimesh',
    'networkx',
    'tabulate',
    'matplotlib',
    'scikit-image',
    'pytorch-memlab',
    'torchmetrics>=0.7',
]

benchmark_requires = [
    'protobuf<4.21',
    'wandb',
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
        'graphgym': graphgym_requires,
        'full': full_requires,
        'benchmark': benchmark_requires,
        'test': test_requires,
        'dev': dev_requires,
    },
    packages=find_packages(),
    include_package_data=True,  # Ensure that `*.jinja` files are found.
)
