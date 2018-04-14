from setuptools import setup, find_packages

__version__ = '1.0.0'
url = 'https://github.com/rusty1s/pytorch_geometric'

print(find_packages())

install_requires = [
    'cffi',
    'torch-scatter',
    'torch-unique',
    'torch-cluster',
    'torch-spline-conv',
    'numpy',
    'scipy',
    'h5py',
    'plyfile',
    'requests',
    'nose',
]
setup_requires = ['pytest-runner']
tests_require = ['pytest', 'pytest-cov']

setup(
    name='torch_geometric',
    version=__version__,
    description='Geometric Deep Learning Extension Library for PyTorch',
    author='Matthias Fey',
    author_email='matthias.fey@tu-dortmund.de',
    url=url,
    download_url='{}/archive/{}.tar.gz'.format(url, __version__),
    keywords=[
        'pytorch', 'geometric-deep-learning', 'graph', 'mesh',
        'neural-networks', 'spline-cnn'
    ],
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    packages=['torch_geometric'])
