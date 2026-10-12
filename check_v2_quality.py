#!/usr/bin/env python3
"""Run v2 architecture gates from a source checkout."""

from __future__ import annotations

from pathlib import Path

from mmtrainer.quality import audit_source_tree


def main() -> int:
    package = Path(__file__).resolve().parent / "src" / "mmtrainer"
    report = audit_source_tree(package)
    if report.violations:
        for violation in report.violations:
            print(violation)
        return 1
    print(
        "training-code-v2-quality: ok "
        f"files={report.files} loc={report.non_test_python_loc} "
        f"argparse_files={report.argparse_files}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
