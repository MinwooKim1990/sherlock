"""Guard: sherlock.__version__ must match pyproject.toml [project].version.

The two drifted for three releases (__version__ stuck at 1.7.0 while the package
shipped 1.10.0), putting the wrong version string in importers' hands. This test
fails CI on any future drift.
"""

import tomllib
from pathlib import Path

import sherlock


def test_dunder_version_matches_pyproject() -> None:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    assert sherlock.__version__ == data["project"]["version"], (
        f"sherlock.__version__={sherlock.__version__!r} != "
        f"pyproject version={data['project']['version']!r}"
    )
