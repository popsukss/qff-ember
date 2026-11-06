"""Cheap checks on the saved submission notebook; never execute its cells."""

from __future__ import annotations

import json
from pathlib import Path
import re

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [ROOT / "notebooks" / "ember.ipynb", ROOT / "notebooks" / "ember_explained.ipynb"]
HEADINGS = [
    "# Ember",
    "## Data and sector geometry",
    "## Quantum reservoir",
    "## Baselines",
    "## Test 2023 results",
    "## Controls",
    "## Real hardware (E6)",
    "## ZF015 / Yellowknife case",
    "## More qubits and inputs (after freezing)",
    "## Conclusions and limits",
]


@pytest.fixture(scope="module", params=NOTEBOOKS, ids=lambda path: path.stem)
def notebook(request):
    assert request.param.is_file()
    value = nbformat.read(request.param, as_version=4)
    nbformat.validate(value)
    value.path = request.param
    return value


def test_notebook_executed_and_small(notebook):
    assert notebook.path.stat().st_size < 20_000_000
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert code_cells
    for cell in code_cells:
        assert isinstance(cell.execution_count, int) and cell.execution_count > 0
        assert all(output.output_type != "error" for output in cell.outputs)


def test_notebook_sources_and_outputs(notebook):
    forbidden = ("QiskitRuntimeService", "save_account", "getpass", "token", "/Users/")
    for cell in notebook.cells:
        content = cell.source + json.dumps(cell.get("outputs", []), ensure_ascii=False)
        for text in forbidden:
            assert text.casefold() not in content.casefold(), (cell.id, text)


def test_notebook_sections_in_order(notebook):
    headings = [line for cell in notebook.cells if cell.cell_type == "markdown"
                for line in cell.source.splitlines() if line.startswith(("# ", "## "))]
    expected = list(HEADINGS)
    if notebook.path.stem == "ember_explained":  # no Controls; multi-qubit results right after the test results
        expected.remove("## Controls")
        expected.remove("## More qubits and inputs (after freezing)")
        expected.insert(expected.index("## Real hardware (E6)"), "## More qubits and inputs (after freezing)")
    assert headings == expected


def test_e6_recomputation_differences_printed(notebook):
    cell = next(cell for cell in notebook.cells if cell.id == "e6-recompute")
    assert "hardware/e6_hw_features.npy" in cell.source
    assert "results/qrc_z_readout.json" in cell.source
    output = "".join(item.get("text", "") for item in cell.outputs if item.output_type == "stream")
    differences = re.findall(r"E6 \+(24|72) h absolute MAE difference: ([0-9.eE+-]+) km", output)
    assert [lead for lead, _ in differences] == ["24", "72"]
    assert all(0 <= float(value) < 1e-9 for _, value in differences)


def test_notebook_is_self_contained(notebook):
    """No repository imports or file reads: every input is embedded."""
    forbidden = ("from ember", "import ember", "os.chdir", "Path.cwd", "read_text(",
                 "open(", "filename=", "sys.path")
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        for text in forbidden:
            assert text not in cell.source, (cell.id, text)


def test_embedded_files_match_repository(notebook):
    import base64
    import gzip
    import hashlib

    cell = next(cell for cell in notebook.cells if cell.id == "embedded-data")
    literal = cell.source.split('BUNDLE = "', 1)[1].rstrip().rstrip('"')
    files = json.loads(gzip.decompress(base64.b64decode(literal)))
    assert files
    for name, item in files.items():
        data = base64.b64decode(item["b64"])
        assert hashlib.sha256(data).hexdigest() == item["sha256"], name
        assert data == (ROOT / name).read_bytes(), f"stale embedded copy: {name}"


def test_reservoir_code_matches_package(notebook):
    cell = next(cell for cell in notebook.cells if cell.id == "reservoir-code")
    package = (ROOT / "ember" / "reservoir.py").read_text()
    assert cell.source.strip() in package
