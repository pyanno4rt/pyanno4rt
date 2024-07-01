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
import inspect
import os
import sys
sys.path.insert(0, os.path.abspath('../'))

# -- Project information -----------------------------------------------------

project = 'pyanno4rt'
copyright = '2024, Karlsruhe Institute of Technology'
author = 'Tim Ortkamp'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
	'myst_nb',
	'sphinx_copybutton',
	'sphinx.ext.autodoc',
	'autoapi.extension',
	'sphinx.ext.napoleon',
	'sphinx.ext.linkcode',
	'sphinx_favicon',
	'sphinx_last_updated_by_git'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'logo_only': True,
    'display_version': True
}
html_logo = '../logo/logo_white.png'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Add the favicons
favicons = [
    'android-chrome-192x192.png',
    'android-chrome-512x512.png',
    'apple-touch-icon.png',
    'favicon.ico',
    'favicon-16x16.png',
    'favicon-32x32.png'
    ]

# -- MystNB options
nb_execution_mode = 'off'

# -- Napoleon options
napoleon_include_init_with_doc = False

# -- AutoAPI configuration ---------------------------------------------------

autoapi_type = 'python'
autoapi_dirs = ['../pyanno4rt']
autoapi_template_dir = '_templates/autoapi'
autoapi_options = [
    'members',
    'undoc-members',
    'show-inheritance',
    'show-module-summary',
    'imported-members',
]

autoapi_ignore = ['*/resources_rc.py', '*/compare_window.py', '*/info_window.py', '*/log_window.py', '*/main_window.py', '*/plan_creation_window.py', '*/settings_window.py', '*/text_window.py', '*/tree_window.py', '*/decision_tree_ntcp_window.py', '*/decision_tree_tcp_window.py', '*/dose_uniformity_window.py', '*/equivalent_uniform_dose_window.py', '*/k_nearest_neighbors_ntcp_window.py', '*/k_nearest_neighbors_tcp_window.py', '*/lkb_ntcp_window.py', '*/logistic_regression_ntcp_window.py', '*/logistic_regression_tcp_window.py', '*/lq_poisson_tcp_window.py', '*/maximum_dvh_window.py', '*/mean_dose_window.py', '*/minimum_dvh_window.py', '*/naive_bayes_ntcp_window.py', '*/naive_bayes_tcp_window.py', '*/neural_network_ntcp_window.py', '*/neural_network_tcp_window.py', '*/random_forest_ntcp_window.py', '*/random_forest_tcp_window.py', '*/squared_deviation_window.py', '*/squared_overdosing_window.py', '*/squared_underdosing_window.py', '*/support_vector_machine_ntcp_window.py', '*/support_vector_machine_tcp_window.py']

# -- Custom auto_summary() macro ---------------------------------------------------

def contains(seq, item):
    """Jinja2 custom test to check existence in a container.

    Example of use:
    {% set class_methods = methods|selectattr("properties", "contains", "classmethod") %}

    Related doc: https://jinja.palletsprojects.com/en/3.1.x/api/#custom-tests
    """
    return item in seq


def prepare_jinja_env(jinja_env) -> None:
    """Add `contains` custom test to Jinja environment."""
    jinja_env.tests["contains"] = contains

autoapi_prepare_jinja_env = prepare_jinja_env

# Options for the linkcode extension
# ----------------------------------
# Resolve function
# This function is used to populate the (source) links in the API
def linkcode_resolve(domain, info):

    import pyanno4rt

    def find_source():
        obj = sys.modules[info['module']]
        for part in info['fullname'].split('.'):
            try:
                obj = getattr(obj, part)
            except:
                return None
        if inspect.isclass(obj) or inspect.isfunction(obj):
            sl = inspect.getsourcelines(obj)
            if sl:
                if obj.__module__ == 'builtins':
                    return obj.__qualname__, sl[1], sl[1] + len(sl[0])-1
                return obj.__module__,  sl[1], sl[1] + len(sl[0])-1
            return None
        else:
            return None

    if domain != 'py' or info['module'] in ['', '_custom_styles']:
        return None

    source = find_source()
    if source:
    	filename = source[0].replace('.', '/') + '.py'
    	return f"https://github.com/pyanno4rt/pyanno4rt/blob/master/{filename}#L{source[1]}-L{source[2]}"
    
    return None

# Custom role for labels used in auto_summary() tables.
rst_prolog = """
.. role:: summarylabel
"""

# Related custom CSS
html_css_files = [
    'css/custom.css',
]
