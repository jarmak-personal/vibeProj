"""Sphinx configuration for vibeProj documentation."""

project = "vibeProj"
copyright = "2025, vibeProj Contributors"
author = "vibeProj Contributors"
release = "0.1.0"

extensions = [
    "myst_parser",
    "sphinx.ext.intersphinx",
]

intersphinx_mapping = {
    "vibespatial": ("https://jarmak-personal.github.io/vibeSpatial/", None),
    "vibespatial-raster": ("https://jarmak-personal.github.io/vibespatial-raster/", None),
}

myst_enable_extensions = [
    "colon_fence",
    "fieldlist",
    "html_admonition",
    "attrs_inline",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

# -- Theme: Furo + vibeSpatial overlay ---------------------------------------
html_theme = "furo"
html_title = "vibeProj"

html_static_path = ["_static"]
html_css_files = ["css/vibespatial.css"]
html_js_files = ["js/vibespatial.js"]

html_theme_options = {
    "source_repository": "https://github.com/vibeProj/vibeProj",
    "source_branch": "main",
    "source_directory": "docs/",
    "light_css_variables": {},  # vibeSpatial CSS forces dark everywhere
    "dark_css_variables": {},
}

# Force dark mode default
html_context = {
    "default_mode": "dark",
}
