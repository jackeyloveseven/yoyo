from pathlib import Path

from check_docs import audit_documents

TRAINING_CODE_ROOT = Path(__file__).resolve().parents[2]


def test_current_documentation_contract() -> None:
    assert audit_documents(TRAINING_CODE_ROOT) == []


def test_mutable_status_and_machine_home_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "CURRENT_STATUS.md").write_text(
        "# status\n\n/home/ubuntu/example\n", encoding="utf-8"
    )
    violations = audit_documents(tmp_path)
    assert "mutable-document:CURRENT_STATUS.md" in violations
    assert "machine-home-path:CURRENT_STATUS.md" in violations


def test_broken_link_and_duplicate_heading_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "# docs\n\n## Same\n\n[missing](missing.md)\n\n## Same\n",
        encoding="utf-8",
    )
    violations = audit_documents(tmp_path)
    assert "broken-local-link:README.md:missing.md" in violations
    assert "duplicate-heading:README.md:same:2" in violations


def test_duplicate_documents_and_skill_root_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "one.md").write_text("# duplicate\n", encoding="utf-8")
    (tmp_path / "two.md").write_text("# duplicate\n", encoding="utf-8")
    (tmp_path / "skills").mkdir()
    violations = audit_documents(tmp_path)
    assert "duplicate-document:one.md,two.md" in violations
    assert "training-code-skill-root:skills" in violations


def test_duplicate_fifty_token_fragment_is_rejected(tmp_path: Path) -> None:
    shared = " ".join(f"token{index}" for index in range(50))
    (tmp_path / "one.md").write_text(f"# one\n\nleft {shared}\n", encoding="utf-8")
    (tmp_path / "two.md").write_text(f"# two\n\nright {shared}\n", encoding="utf-8")
    violations = audit_documents(tmp_path)
    assert "duplicate-fragment:one.md:two.md" in violations
