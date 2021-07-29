import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="graphgym",
    version="0.3.1",
    author="Jiaxuan You",
    author_email="jiaxuan@cs.stanford.edu",
    description="GraphGym: platform for designing and evaluating Graph Neural Networks (GNN)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/snap-stanford/graphgym",
    packages=setuptools.find_packages(),
    install_requires=[
        'yacs',
        'tensorboardx',
        'torch',
        'torch-geometric',
        'networkx',
        'numpy'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
