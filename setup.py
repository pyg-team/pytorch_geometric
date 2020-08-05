from setuptools import setup, find_packages

__version__ = '1.6.1'
url = 'https://github.com/rusty1s/pytorch_geometric'

install_requires = [
    'torch',
    'numpy',
    'tqdm',
    'scipy',
    'networkx',
    'scikit-learn',
    'numba',
    'requests',
    'pandas',
    'rdflib',
    'h5py',
    'googledrivedownloader',
    'ase',
    'jinja2',
]
setup_requires = ['pytest-runner']
tests_require = ['pytest', 'pytest-cov', 'mock']

setup(
    name='torch_geometric',
    version=__version__,
    description='Geometric Deep Learning Extension Library for PyTorch',
    author='Matthias Fey',
    author_email='matthias.fey@tu-dortmund.de',
    url=url,
    download_url='{}/archive/{}.tar.gz'.format(url, __version__),
    keywords=[
        'pytorch',
        'geometric-deep-learning',
        'graph-neural-networks',
    ],
    python_requires='>=3.6',
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    extras_require={'test': tests_require},
    packages=find_packages(),
    include_package_data=True,
)
