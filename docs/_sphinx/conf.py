# SignalForge — Sphinx configuration

project = "SignalForge"
author = "Shun Richard Honda"
copyright = "2026, Adelic AI"

extensions = [
    "myst_parser",           # Read .md files
    "sphinx.ext.autodoc",    # Auto-generate from docstrings
    "sphinx.ext.napoleon",   # Google/NumPy-style docstrings
    "sphinx.ext.viewcode",   # [source] links to code
    "sphinx.ext.intersphinx",  # Link to numpy/python docs
]

# MyST: allow markdown docs
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Theme
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 3,
    "titles_only": False,
}

# Auto-doc settings
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Intersphinx: link to external docs
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# Paths
templates_path = ["_templates"]
exclude_patterns = ["_build", "archive"]
