# Research Paper Intelligence

A production-oriented Python project for ingesting, processing, and analyzing research papers.

Repository layout (top-level):

- src/                      # Source code package
- data/
  - raw/                    # Raw, unprocessed inputs (do not commit large files)
  - processed/              # Cleaned/processed datasets
- graphs/                   # Generated charts and figures
- reports/                  # Generated reports (PDF, HTML, etc.)
- tests/                    # Unit and integration tests
- logs/                     # Runtime logs

Guiding principles

- Use src/ layout to avoid accidental imports from the repository root.
- Keep raw and generated artifacts out of VCS; track small placeholders with .gitkeep.
- Favor small, focused modules and typed interfaces for maintainability and testability.

Getting started

1. Create a virtual environment and install runtime dependencies:

   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt

2. Run tests:

   python -m pytest -q

3. Run the package without installing (example):

   python run.py ingest file:///absolute/path/to/paper.json

Contributing

See CONTRIBUTING.md (not yet added) for guidelines.
