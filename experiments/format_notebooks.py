#!/usr/bin/env python3
"""
Format all .ipynb files in the project with 1-space JSON indentation.
"""

import os
import nbformat


def format_notebook(path):
    with open(path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    with open(path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f, indent=1)  # Jupyter's default indentation


for root, _, files in os.walk('.'):
    for file in files:
        if file.endswith('.ipynb'):
            path = os.path.join(root, file)
            print(f'Formatting {path}')
            format_notebook(path)
