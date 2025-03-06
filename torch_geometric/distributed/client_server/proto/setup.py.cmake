from setuptools import setup, find_packages

setup(
    name='client_server_proto',
    version='0.0.0',
    install_requires=['grpcio', 'protobuf', 'wheel', 'typing_extensions'],
    package_data={'client_server_proto': ['py.typed', 'client_server_proto/*.py']},
    python_requires='>=3.8',
)
