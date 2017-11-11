#!/bin/sh

git checkout gh-pages
git checkout master docs torch_geometric
cd docs && make html
