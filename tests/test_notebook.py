"""Cheap checks on the saved submission notebook; never execute its cells."""

from __future__ import annotations

import json
from pathlib import Path
import re

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "ember.ipynb"
HEADINGS = [
    "# Ember",
    "## Data and sector geometry",
    "## Quantum reservoir",
    "## Baselines",
    "## Test 2023 results",
    "## Controls",
    "## Real hardware (E6)",
    "## ZF015 / Yellowknife case",
    "## Conclusions and limits",
]


@pytest.fixture(scope="module")
def notebook():
    assert NOTEBOOK.is_file()
    value = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(value)
    return value


def test_notebook_executed_and_small(notebook):
    assert NOTEBOOK.stat().st_size < 20_000_000
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
    assert headings == HEADINGS


def test_e6_recomputation_differences_printed(notebook):
    cell = next(cell for cell in notebook.cells if cell.id == "e6-recompute")
    assert "hardware/e6_hw_features.npy" in cell.source
    assert "results/qrc_z_readout.json" in cell.source
    output = "".join(item.get("text", "") for item in cell.outputs if item.output_type == "stream")
    differences = re.findall(r"E6 \+(24|72) h absolute MAE difference: ([0-9.eE+-]+) km", output)
    assert [lead for lead, _ in differences] == ["24", "72"]
    assert all(0 <= float(value) < 1e-9 for _, value in differences)
