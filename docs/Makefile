# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = pytorch_geometric
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

gh-pages:
	git checkout gh-pages
	cd ..
	git checkout master docs torch_geometric
	cd docs
	make html
	cd ..
	mv -fv build/html/* ./
	rm -rf docs torch_geometric
	git add -A
	git commit -m 'Generated gh-pages for `git log master -1 --pretty=short --abbrev-commit`'
	git push origin gh-pages
	git checkout master
	cd docs
