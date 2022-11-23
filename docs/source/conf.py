# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))
# sys.path.append('/Users/caitlinjones/Documents/Documents_new/QuTAC/Generative/QML-QUTAC/qugen')

project = 'qugen'
copyright = '2023 QUTAC, BASF Digital Solutions GmbH, BMW Group, Lufthansa Industry Solutions AS GmbH, Merck KGaA (Darmstadt, Germany), Munich Re, SAP SE.'
author = 'Joseph Doetsch, Thomas Ehmer, Caitlin Jones, Florian Krellner, Johannes Klepsch, Andre Luckow, Oliver Mitevski, Carlos A. Riofrío, Aleksandar Vučković'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'm2r2',
    "sphinx_immaterial"
]

source_suffix = [".rst", ".md"]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_immaterial'
html_static_path = ['_static']
html_theme_options = {
    "rightsidebar": "false",
    "sidebarwidth": "400",
    "palette": { "primary": "indigo" }
}
html_sidebars = { '**': ['globaltoc.html', 'relations.html', 'sourcelink.html', 'searchbox.html'] }
html_logo = "logo.jpeg"

