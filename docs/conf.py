"""Sphinx configuration file."""

import geomfum

project = "GeomFuM"
copyright = "2025, GeomFuM contributors"
author = "GeomFuM Team"
release = version = getattr(geomfum, "__version__", "latest")
html_static_path = ["_static"]
extensions = [
    "nbsphinx",
    "nbsphinx_link",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
    "sphinx.ext.viewcode",
    "sphinx_gallery.load_style",
    "sphinx_copybutton",
    "sphinx_design",
]

autosummary_generate = True

# Configure napoleon for numpy docstring
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = False
napoleon_use_ivar = False
napoleon_use_rtype = False
napoleon_include_init_with_doc = False

# Configure nbsphinx for notebooks execution
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]
nbsphinx_execute = "never"  # Don't execute notebooks during build
nbsphinx_input_prompt = " In [%s]:"
nbsphinx_output_prompt = " Out [%s]:"
nbsphinx_allow_errors = True

# Notebook paths
nbsphinx_prolog = r"""
{% set docname = env.doc2path(env.docname, base=None) %}

.. raw:: html

    <div class="admonition note">
      <p>Notebook source code:
        <a class="reference external" href="https://github.com/3diglab/geomfum/main/{{ docname|e }}">{{ docname|e }}</a>
        <br>Run it yourself on binder
        <a href="https://mybinder.org/v2/gh/3diglab/geomfum/main?filepath={{ docname|e }}"><img alt="Binder badge"
        src="https://mybinder.org/badge_logo.svg"
        style="vertical-align:text-bottom"></a>
      </p>
    </div>

.. raw:: latex

    \nbsphinxstartnotebook{\scriptsize\noindent\strut
    \textcolor{gray}{The following section was generated from
    \sphinxcode{\sphinxupquote{\strut {{ docname | escape_latex }}}} \dotfill}}
    """

nbsphinx_thumbnails = {}

# Include notebooks directory
templates_path = ["_templates"]
source_suffix = [".rst"]
main_doc = "index"
language = "en"

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    "notebooks/.ipynb_checkpoints",
    "notebooks/*/.ipynb_checkpoints",
    "notebooks/demos/**",  # Exclude demos to avoid large files
]

bibtex_bibfiles = ["references.bib"]

pygments_style = None

html_theme = "pydata_sphinx_theme"
html_logo = "../GeomFuMlogo.png"
html_favicon = "../GeomFuMlogo_only.png"
html_static_path = ["_static"]
html_baseurl = "geomfum.github.io"
htmlhelp_basename = "geomfumdoc"
html_last_updated_fmt = "%c"

# GitHub context for edit page button
html_context = {
    "github_user": "3diglab",
    "github_repo": "geomfum",
    "github_version": "main",
    "doc_path": "docs",
}

# PyData theme options
html_theme_options = {
    "github_url": "https://github.com/3diglab/geomfum",
    "use_edit_page_button": False,
    "show_toc_level": 2,
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "icon_links": [
        {
            "name": "Discord",
            "url": "https://discord.gg/THHku2ckJs",
            "icon": "fab fa-discord",
        },
    ],
}

latex_elements = {}
latex_documents = [
    (
        main_doc,
        "geomfum.tex",
        "GeomFuM Documentation",
        "GeomFuM Team",
        "manual",
    ),
]
man_pages = [(main_doc, "geomfum", "GeomFuM Documentation", [author], 1)]
texinfo_documents = [
    (
        main_doc,
        "geomfum",
        "GeomFuM Documentation",
        author,
        "geomfum",
        "One line description of project.",
        "Miscellaneous",
    ),
]
epub_title = project
epub_exclude_files = ["search.html"]

# -- intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "joblib": ("https://joblib.readthedocs.io/en/latest", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/dev", None),
    "python": ("https://docs.python.org/3", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}
intersphinx_timeout = 6

# Handle import errors gracefully
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "imported-members": False,
}

# Suppress import warnings for circular dependencies during doc build
# Suppress duplicate object description warnings
suppress_warnings = [
    "app.add_node",
    "app.add_directive",
    "app.add_role",
    "autosummary",
    "ref.ref",
]
nitpicky = False
