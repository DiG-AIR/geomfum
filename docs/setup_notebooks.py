#!/usr/bin/env python3
"""Set up notebook structure in the docs directory. This creates symbolic links to the actual notebooks in the project root."""

import shutil
from pathlib import Path


def setup_notebooks():
    """Set up notebook structure in docs directory."""
    # Get the project root and docs directory
    project_root = Path(__file__).parent.parent
    docs_dir = Path(__file__).parent

    # Create notebooks directory in docs
    notebooks_dir = docs_dir / "notebooks"
    notebooks_dir.mkdir(exist_ok=True)

    # Create how_to and demos subdirectories
    how_to_dir = notebooks_dir / "how_to"
    demos_dir = notebooks_dir / "demos"
    how_to_dir.mkdir(exist_ok=True)
    demos_dir.mkdir(exist_ok=True)

    # Source notebook directories
    src_notebooks = project_root / "notebooks"
    src_how_to = src_notebooks / "how_to"
    src_demos = src_notebooks / "demos"

    # Copy how_to notebooks
    if src_how_to.exists():
        for notebook in src_how_to.glob("*.ipynb"):
            dest = how_to_dir / notebook.name
            if dest.exists():
                dest.unlink()
            shutil.copy2(notebook, dest)
            print(f"Copied: {notebook.name}")

    # Copy demo notebooks
    if src_demos.exists():
        for notebook in src_demos.glob("*.ipynb"):
            dest = demos_dir / notebook.name
            if dest.exists():
                dest.unlink()
            shutil.copy2(notebook, dest)
            print(f"Copied: {notebook.name}")

    print("Notebook setup complete!")


if __name__ == "__main__":
    setup_notebooks()
