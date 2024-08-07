# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import sphinx_theme_pd

sys.path.insert(0, os.path.abspath("../../src"))


# -- Project information -----------------------------------------------------

project = "Partitioning"
copyright = "2024, Einara Zahn"
author = "Einara Zahn"

# The full version, including alpha/beta/rc tags
release = "1.0.1"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # For Google style and NumPy style docstrings
    "sphinx.ext.viewcode",  # Optional, for viewing source code
    # "sphinx_rtd_theme",  # If you are using the Read the Docs theme
]

todo_include_todos = True

# html_theme = 'sphinx_pdj_theme'
# html_theme_path = [sphinx_pdj_theme.get_html_theme_path()]

html_theme = "sphinx_theme_pd"
html_theme_path = [sphinx_theme_pd.get_html_theme_path()]

# import sphinx_pdj_theme
# html_theme = 'sphinx_pdj_theme'
# html_theme_path = [sphinx_pdj_theme.get_html_theme_path()]

# html_static_path = ["_static"]
# html_css_files = [
#    "css/custom.css",
# ]
# html_theme_options = {
#    "collapse_navigation": False,
#    "sticky_navigation": True,
#    "navigation_depth": 4,
#    "includehidden": True,
#    "titles_only": False,
# }

autodoc_default_options = {
    "members": True,
    "special-members": "__call__",
    "inherited-members": True,
    "show-inheritance": True,
    "exclude-members": "__init__",
}

# autodoc_default_options = {
#    'members': True,
#    'undoc-members': True,
#    'special-members': '__call__',
#    'private-members': True,
#    'inherited-members': True,
#    'show-inheritance': True,
#    'ignore-module-all': True,
#    'exclude-members': '__init__'
# }

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.6", None),
    "pandas": ("http://pandas.pydata.org/pandas-docs/stable/", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = "alabaster"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
