import datetime
import os.path as osp
import sys

import pyg_sphinx_theme

import torch_geometric

author = 'PyG Team'
project = 'pytorch_geometric'
version = torch_geometric.__version__
copyright = f'{datetime.datetime.now().year}, {author}'

sys.path.append(osp.join(osp.dirname(pyg_sphinx_theme.__file__), 'extension'))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'pyg',
]

html_theme = 'pyg_sphinx_theme'
html_logo = ('https://raw.githubusercontent.com/pyg-team/pyg_sphinx_theme/'
             'master/pyg_sphinx_theme/static/img/pyg_logo.png')
html_favicon = ('https://raw.githubusercontent.com/pyg-team/pyg_sphinx_theme/'
                'master/pyg_sphinx_theme/static/img/favicon.png')
html_static_path = ['_static']
templates_path = ['_templates']

add_module_names = False
autodoc_member_order = 'bysource'

suppress_warnings = ['autodoc.import_object']

intersphinx_mapping = {
    'python': ('https://docs.python.org/', None),
    'numpy': ('http://docs.scipy.org/doc/numpy', None),
    'pandas': ('http://pandas.pydata.org/pandas-docs/dev', None),
    'torch': ('https://pytorch.org/docs/master', None),
}


def setup(app):
    def rst_jinja_render(app, _, source):
        rst_context = {'torch_geometric': torch_geometric}
        source[0] = app.builder.templates.render_string(source[0], rst_context)

    app.connect('source-read', rst_jinja_render)
    app.add_js_file('js/version_alert.js')
