"""Rasterize the standalone architecture diagram for the README.

    cd docs/report && pdflatex architecture_standalone.tex
    python docs/scripts/render_architecture.py

Writes docs/figures/architecture.png from docs/report/architecture_standalone.pdf.
Requires PyMuPDF (pip install pymupdf); it is a documentation tool, not an inference dependency.
"""
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "docs" / "report" / "architecture_standalone.pdf"
TARGET = ROOT / "docs" / "figures" / "architecture.png"


def main() -> None:
    with pymupdf.open(SOURCE) as document:
        pixmap = document[0].get_pixmap(dpi=230, alpha=False)
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        pixmap.save(TARGET)
    print("wrote", TARGET)


if __name__ == "__main__":
    main()
