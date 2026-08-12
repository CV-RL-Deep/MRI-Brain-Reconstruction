#!/usr/bin/env python3
"""
Project Dumper for LLM Vibecoding
Dumps a Python project into a single, well-formatted Markdown file.
"""

import json
import subprocess
import sys

from pathlib import Path
from typing import List


# --- Configuration ---

# Extensions to completely ignore (binaries, caches, etc. to save tokens)
IGNORE_EXTENSIONS = {
    '.pyc',
    '.pyo',
    '.so',
    '.dll',
    '.dylib',
    '.bin',
    '.exe',
    '.png',
    '.jpg',
    '.jpeg',
    '.gif',
    '.svg',
    '.ico',
    '.pdf',
    '.zip',
    '.tar',
    '.gz',
    '.bz2',
    '.7z',
    '.sqlite',
    '.db',
    '.woff',
    '.woff2',
    '.ttf',
    '.eot',
    '.md',
}

# Directories to ignore if falling back to os.walk (not in a git repo)
IGNORE_DIRS = {
    '.git',
    '__pycache__',
    'venv',
    '.venv',
    'env',
    '.env',
    'node_modules',
    '.mypy_cache',
    '.pytest_cache',
    '.tox',
    'build',
    'dist',
    '.eggs',
    '*.egg-info',
}

# Map file extensions to Markdown syntax highlighting
LANG_MAP = {
    '.py': 'python',
    '.txt': 'text',
    '.yaml': 'yaml',
    '.yml': 'yaml',
    '.toml': 'toml',
    '.json': 'json',
    '.md': 'markdown',
    '.sh': 'bash',
    '.bash': 'bash',
    '.dockerfile': 'dockerfile',
    '.cfg': 'ini',
    '.ini': 'ini',
    '.html': 'html',
    '.css': 'css',
    '.js': 'javascript',
    '.ts': 'typescript',
    '.sql': 'sql',
    '.xml': 'xml',
    '.csv': 'csv',
}


def get_lang(filepath: Path) -> str:
    """Determine markdown language tag for syntax highlighting."""
    name = filepath.name.lower()
    if name.startswith('dockerfile'):
        return 'dockerfile'
    return LANG_MAP.get(filepath.suffix.lower(), 'text')


def get_project_files(root_dir: Path) -> List[Path]:
    """Get list of project files, preferring git tracked files."""
    try:
        # Use git to get tracked files. -z handles filenames with spaces/newlines.
        result = subprocess.run(
            ['git', 'ls-files', '-z'],
            cwd=root_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        files = [f for f in result.stdout.split('\0') if f]
        return [root_dir / f for f in files]
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(
            "Warning: Not a git repository or git not found. Falling back to directory walk."
        )
        files = []
        for p in root_dir.rglob('*'):
            if p.is_file() and not any(
                part in IGNORE_DIRS or part.startswith('.') for part in p.parts
            ):
                # Fix for the startswith('.') ignoring hidden files like .gitignore.
                # We only want to ignore hidden *directories*.
                pass

        # Better fallback logic for hidden files vs hidden dirs
        files = []
        for p in root_dir.rglob('*'):
            if p.is_file():
                # Check if any parent directory is in ignore list
                if any(part in IGNORE_DIRS for part in p.parts):
                    continue
                files.append(p)
        return files


def categorize_files(files: List[Path]) -> dict:
    """Categorize files into Functional, Docker, Aux, Config, and Other."""
    categories = {
        'functional': [],
        'docker': [],
        'aux': [],
        'config': [],
        'notebooks': [],
        'other': [],
    }

    for f in files:
        if f.suffix.lower() in IGNORE_EXTENSIONS:
            continue

        name = f.name.lower()
        ext = f.suffix.lower()

        # 2. Docker files
        if (
            name.startswith('dockerfile')
            or name.startswith('docker-compose')
            or ext == '.dockerfile'
        ):
            categories['docker'].append(f)
        # 1. Functional Python/Env files
        elif (
            ext == '.py'
            or name.startswith('requirements')
            or ext in ('.yaml', '.yml')
        ):
            categories['functional'].append(f)
        # 3. Auxiliary files
        elif name in (
            'pyproject.toml',
            'setup.py',
            'setup.cfg',
            '.gitignore',
            '.gitattributes',
            'manifest.in',
            'tox.ini',
            '.flake8',
            '.pre-commit-config.yaml',
            'makefile',
        ):
            categories['aux'].append(f)
        # 4. Config files (JSON, .json.base, etc.)
        elif ext == '.json' or '.json' in name or ext in ('.ini', '.cfg'):
            categories['config'].append(f)
        # 5. Jupyter Notebooks
        elif ext == '.ipynb':
            categories['notebooks'].append(f)
        else:
            categories['other'].append(f)

    # Sort files within categories for consistent output
    for cat in categories:
        categories[cat].sort(key=lambda x: x.name)

    return categories


def generate_tree(root_dir: Path, files: List[Path]) -> str:
    """Generate a visual directory tree string."""
    tree = {}
    for f in files:
        if f.suffix.lower() in IGNORE_EXTENSIONS and f.suffix.lower() != '.md':
            continue
        rel = f.relative_to(root_dir)
        parts = rel.parts
        current = tree
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = None

    lines = [f"{root_dir.resolve().name}/"]

    def walk(node, prefix=""):
        # Sort: directories first, then files, alphabetically
        items = sorted(node.items(), key=lambda x: (x[1] is None, x[0]))
        for i, (name, sub) in enumerate(items):
            is_last = i == len(items) - 1
            connector = '└── ' if is_last else '├── '
            lines.append(f'{prefix}{connector}{name}')
            if sub is not None:
                extension = '    ' if is_last else '│   '
                walk(sub, prefix + extension)

    if tree:
        walk(tree)
    return '\n'.join(lines)


def read_file_safe(filepath: Path, max_size_kb: int) -> str:
    """Read file content safely, handling encodings and size limits."""
    size_kb = filepath.stat().st_size / 1024
    if size_kb > max_size_kb:
        return f"[SKIPPED: File size ({size_kb:.1f} KB) exceeds limit of {max_size_kb} KB to save context tokens]"

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(filepath, 'r', encoding='latin-1') as f:
                return f.read()
        except Exception:
            return "[SKIPPED: Binary or unreadable file]"
    except Exception as e:
        return f"[ERROR reading file: {e}]"


def convert_notebook_to_percent(filepath: Path) -> str:
    """Converts a Jupyter Notebook (.ipynb) to percent-format Python script."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except Exception as e:
        return f"[ERROR reading notebook: {e}]"

    lines = []
    for cell in nb.get('cells', []):
        cell_type = cell.get('cell_type', '')
        source = cell.get('source', [])

        # Source can be a list of strings or a single string depending on the notebook version
        if isinstance(source, list):
            source_text = "".join(source)
        else:
            source_text = str(source)

        if cell_type == 'code':
            lines.append("# %%")
            lines.append(source_text.rstrip('\n'))
        elif cell_type == 'markdown':
            lines.append("# %% [markdown]")
            for line in source_text.splitlines():
                # Avoid trailing spaces on empty markdown lines
                lines.append(f"# {line}" if line else "#")
        else:
            lines.append(f"# %% [{cell_type}]")
            lines.append(source_text.rstrip('\n'))

    return "\n".join(lines)


def dump_project(root_dir: Path, output_file: Path, args):
    """Main dumping logic."""
    print(f"Scanning project at: {root_dir.resolve()}")
    all_files = get_project_files(root_dir)

    # Exclude this script from the dump
    script_name = Path(__file__).name
    all_files = [
        f for f in all_files if f.name not in (output_file.name, script_name)
    ]

    if not all_files:
        print("No files found. Exiting.")
        sys.exit(1)

    categories = categorize_files(all_files)

    # Filter out optional categories based on args
    if not args.include_docker:
        categories['docker'] = []
    if not args.include_other:
        categories['other'] = []
    if args.exclude_aux:
        categories['aux'] = []
    if args.exclude_config:
        categories['config'] = []

    # Count total files to be dumped
    total_files = sum(len(v) for v in categories.values())
    print(
        f"Found {len(all_files)} total files. Dumping {total_files} relevant files..."
    )

    with open(output_file, 'w', encoding='utf-8') as md:
        # 0. Header and Tree
        md.write(f"# Project Dump: {root_dir.resolve().name}\n\n")
        md.write("## Project Structure\n\n")
        md.write("```text\n")
        md.write(generate_tree(root_dir, all_files))
        md.write("\n```\n\n")
        md.write("---\n\n")

        # Define the order and titles for the sections
        sections = [
            ('functional', '1. Functional Code & Dependencies'),
            ('docker', '2. Docker & Compose Files'),
            ('aux', '3. Auxiliary & Build Files'),
            ('config', '4. Configuration Files'),
            ('notebooks', '5. Jupyter Notebooks (Percent Format)'),
            ('other', '5. Other Files'),
        ]

        for cat_key, section_title in sections:
            files_in_cat = categories[cat_key]
            if not files_in_cat:
                continue

            md.write(f"## {section_title}\n\n")

            for filepath in files_in_cat:
                rel_path = filepath.relative_to(root_dir)
                lang = get_lang(filepath)

                # Convert Jupyter Notebooks to percent-format Python scripts
                if filepath.suffix.lower() == '.ipynb':
                    content = convert_notebook_to_percent(filepath)
                    lang = 'python'  # force python syntax highlighting for notebooks
                else:
                    content = read_file_safe(filepath, args.max_file_size)

                md.write(f"### `{rel_path}`\n\n")
                md.write(f"```{lang}\n")
                md.write(content)
                if not content.endswith("\n"):
                    md.write("\n")
                md.write("```\n\n")
                md.write("---\n\n")

    print(f"Successfully dumped project to: {output_file.resolve()}")
    print(f"Total size: {output_file.stat().st_size / 1024:.1f} KB")


def main(args):
    root_path = Path(args.root).resolve()
    if not root_path.is_dir():
        print(f"Error: {root_path} is not a valid directory.")
        sys.exit(1)

    dump_project(root_path, Path(args.output), args)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Dump a Python project into a single Markdown file for LLM context."
    )
    parser.add_argument(
        '--root',
        default='.',
        help="Project root directory (default: current dir)",
    )
    parser.add_argument(
        '--output',
        default='project-dump.md',
        help="Output markdown file (default: project-dump.md)",
    )
    parser.add_argument(
        '--include-docker',
        action='store_true',
        help="Include Docker and compose files (default: off)",
    )
    parser.add_argument(
        '--exclude-aux',
        action='store_true',
        help="Exclude auxiliary files like pyproject.toml, .gitignore (default: included)",
    )
    parser.add_argument(
        '--exclude-config',
        action='store_true',
        help="Exclude JSON and config files (default: included)",
    )
    parser.add_argument(
        '--include-other',
        action='store_true',
        help="Include 'Other' files like bash scripts (default: excluded)",
    )
    parser.add_argument(
        '--max-file-size',
        type=int,
        default=100,
        help="Max file size in KB to include, prevents context overflow (default: 100)",
    )

    args = parser.parse_args()

    main(args)
