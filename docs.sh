#!/bin/sh

git diff-index --quiet HEAD -- || exit 1

git checkout gh-pages
git checkout master docs torch_geometric
cd docs && make html
cd ..
mv -fv docs/build/html/* ./
rm -rf docs torch_geometric
git add -A
git commit -m "Generated gh-pages for $(git log master -1 --pretty=short --abbrev-commit)"
git push origin gh-pages
git checkout master
