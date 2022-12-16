from setuptools import find_packages, setup

__version__ = '2.2.0'
URL = 'https://github.com/pyg-team/pytorch_geometric'

install_requires = [
    'tqdm',
    'numpy',
    'scipy',
    'jinja2',
    'requests',
    'pyparsing',
    'torchmetrics',
    'scikit-learn',
    'psutil>=5.8.0',
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
]

benchmark_requires = [
    'protobuf<4.21',
    'wandb',
    'pandas',
    'matplotlib',
]

test_requires = [
    'pytest',
    'pytest-cov',
    'onnx',
    'onnxruntime',
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
