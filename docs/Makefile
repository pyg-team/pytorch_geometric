SPHINXBUILD   = sphinx-build
SPHINXPROJ    = pytorch_geometric
SOURCEDIR     = source
BUILDDIR      = build

.PHONY: help Makefile

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(0)
