from __future__ import annotations

from pathlib import Path

from mmtrainer.quality import audit_source_tree


def test_v2_source_passes_architecture_gates() -> None:
    source = Path(__file__).resolve().parents[2] / "src" / "mmtrainer"
    report = audit_source_tree(source)
    assert report.violations == ()
    assert report.argparse_files == 1
    assert report.non_test_python_loc <= 103_748
