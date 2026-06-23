# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))


project = 'baorecon'
copyright = '2024, Edoardo Maragliano'
author = 'Edoardo Maragliano'
release = '0.2.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',  # For Google or NumPy style docstrings
]

templates_path = ['_templates']
exclude_patterns = []
# Includere le classi e i membri (metodi e attributi)
autodoc_default_options = {
    'members': True,  # Mostra membri (funzioni, classi, ecc.)
    'undoc-members': True,  # Includi anche membri senza docstring (opzionale)
    'private-members': True,  # Includi membri privati (opzionale)
    'special-members': '__init__',  # Includi metodi speciali, come __init__
    'inherited-members': True,  # Mostra membri ereditati
    'show-inheritance': True,  # Mostra la gerarchia di ereditarietà
}


language = 'python'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
