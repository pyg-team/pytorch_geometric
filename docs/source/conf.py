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
    'sphinx_copybutton',
    'nbsphinx',
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
    # 'numpy': ('http://docs.scipy.org/doc/numpy', None),
    'pandas': ('http://pandas.pydata.org/pandas-docs/dev', None),
    'torch': ('https://pytorch.org/docs/main', None),
}

nbsphinx_thumbnails = {
    'tutorial/create_gnn':
    '_static/thumbnails/create_gnn.png',
    'tutorial/heterogeneous':
    '_static/thumbnails/heterogeneous.png',
    'tutorial/create_dataset':
    '_static/thumbnails/create_dataset.png',
    'tutorial/load_csv':
    '_static/thumbnails/load_csv.png',
    'tutorial/neighbor_loader':
    '_static/thumbnails/neighbor_loader.png',
    'tutorial/point_cloud':
    '_static/thumbnails/point_cloud.png',
    'tutorial/explain':
    '_static/thumbnails/explain.png',
    'tutorial/shallow_node_embeddings':
    '_static/thumbnails/shallow_node_embeddings.png',
    'tutorial/distributed_pyg':
    '_static/thumbnails/distributed_pyg.png',
    'tutorial/multi_gpu_vanilla':
    '_static/thumbnails/multi_gpu_vanilla.png',
    'tutorial/multi_node_multi_gpu_vanilla':
    '_static/thumbnails/multi_gpu_vanilla.png',
}


def rst_jinja_render(app, _, source):
    if hasattr(app.builder, 'templates'):
        rst_context = {'torch_geometric': torch_geometric}
        source[0] = app.builder.templates.render_string(source[0], rst_context)


def setup(app):
    app.connect('source-read', rst_jinja_render)
    app.add_js_file('js/version_alert.js')
