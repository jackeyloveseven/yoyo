#!/usr/bin/env python3
"""Audit the maintained Training Code documentation surface."""

from __future__ import annotations

import argparse
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path

DOCUMENT_SUFFIXES = frozenset({".adoc", ".md", ".mdx", ".rst", ".txt"})
EXCLUDED_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        ".github",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "SimpleTuner",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "source_snapshots",
        "third_party",
    }
)
IMPORTED_SOURCE_ROOTS = frozenset({"gemma4_vlm", "krea2_edit_runtime", "yuzihao_diffsynth"})
FORBIDDEN_BASENAMES = frozenset({"AGENT_HANDOFF.md", "CURRENT_STATUS.md", "TODO.md"})
FORBIDDEN_HOME_PATH = re.compile(r"/(?:home/(?:jackeylove|ubuntu)|Users)/")
INLINE_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
REFERENCE_LINK = re.compile(r"^\s*\[[^\]]+\]:\s*(\S+)", re.MULTILINE)
SECOND_LEVEL_HEADING = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
FRONTMATTER = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
WORD_TOKEN = re.compile(r"[A-Za-z0-9_\u4e00-\u9fff]+")

MAX_DOCUMENTS = 65
MAX_TOTAL_LINES = 2_000
MAX_FILE_LINES = 500
DUPLICATE_TOKEN_WINDOW = 50


def _is_document(path: Path) -> bool:
    return path.suffix.lower() in DOCUMENT_SUFFIXES or path.name == "README"


def _is_excluded(root: Path, path: Path) -> bool:
    relative = path.relative_to(root)
    if EXCLUDED_DIRECTORY_NAMES & set(relative.parts[:-1]) or any(
        part.endswith(".egg-info") for part in relative.parts[:-1]
    ):
        return True
    return (
        len(relative.parts) > 2
        and relative.parts[0] == "imported"
        and relative.parts[1] in IMPORTED_SOURCE_ROOTS
    )


def document_paths(root: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and _is_document(path) and not _is_excluded(root, path)
        )
    )


def _normalized_document(text: str) -> str:
    text = FRONTMATTER.sub("", text)
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def _token_windows(text: str) -> set[tuple[str, ...]]:
    tokens = WORD_TOKEN.findall(CODE_FENCE.sub(" ", text).casefold())
    return {
        tuple(tokens[index : index + DUPLICATE_TOKEN_WINDOW])
        for index in range(max(0, len(tokens) - DUPLICATE_TOKEN_WINDOW + 1))
    }


def _local_link_target(path: Path, raw_target: str) -> Path | None:
    target = raw_target.strip().strip("<>").split(maxsplit=1)[0]
    if target.startswith(("#", "data:", "http://", "https://", "mailto:")):
        return None
    target = target.split("#", 1)[0].split("?", 1)[0]
    if not target or "{" in target or "}" in target:
        return None
    candidate = path.parent / target
    if not candidate.exists() and re.search(r":\d+$", str(candidate)):
        candidate = Path(re.sub(r":\d+$", "", str(candidate)))
    return candidate


def _audit_document(
    root: Path, path: Path
) -> tuple[list[str], int, str | None, set[tuple[str, ...]]]:
    relative = path.relative_to(root).as_posix()
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = len(text.splitlines())
    violations: list[str] = []

    if path.name in FORBIDDEN_BASENAMES:
        violations.append(f"mutable-document:{relative}")
    if lines > MAX_FILE_LINES:
        violations.append(f"document-lines:{relative}:{lines}>{MAX_FILE_LINES}")
    if FORBIDDEN_HOME_PATH.search(text):
        violations.append(f"machine-home-path:{relative}")
    if path.name.startswith("RELEASE_1_") and re.search(r"\]\([^)]*release-manifest\.json", text):
        violations.append(f"mutable-v1-release-manifest-link:{relative}")

    headings = [heading.strip().casefold() for heading in SECOND_LEVEL_HEADING.findall(text)]
    for heading, count in Counter(headings).items():
        if count > 1:
            violations.append(f"duplicate-heading:{relative}:{heading}:{count}")

    for raw_target in INLINE_LINK.findall(text) + REFERENCE_LINK.findall(text):
        target = _local_link_target(path, raw_target)
        if target is not None and not target.exists():
            violations.append(f"broken-local-link:{relative}:{raw_target}")

    normalized = _normalized_document(text)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest() if normalized else None
    return violations, lines, digest, _token_windows(text)


def audit_documents(
    root: Path,
    *,
    max_documents: int = MAX_DOCUMENTS,
    max_total_lines: int = MAX_TOTAL_LINES,
) -> list[str]:
    root = root.resolve(strict=True)
    paths = document_paths(root)
    violations: list[str] = []

    if len(paths) > max_documents:
        violations.append(f"document-count:{len(paths)}>{max_documents}")

    total_lines = 0
    hashes: dict[str, list[Path]] = defaultdict(list)
    fragments: dict[tuple[str, ...], Path] = {}
    for path in paths:
        document_violations, lines, digest, windows = _audit_document(root, path)
        violations.extend(document_violations)
        total_lines += lines
        if digest is not None:
            hashes[digest].append(path)
        for window in windows:
            previous = fragments.setdefault(window, path)
            if previous != path:
                left = previous.relative_to(root).as_posix()
                right = path.relative_to(root).as_posix()
                violations.append(f"duplicate-fragment:{left}:{right}")

    if total_lines > max_total_lines:
        violations.append(f"document-lines-total:{total_lines}>{max_total_lines}")

    for duplicates in hashes.values():
        if len(duplicates) > 1:
            names = ",".join(path.relative_to(root).as_posix() for path in duplicates)
            violations.append(f"duplicate-document:{names}")

    if (root / "skills").exists():
        violations.append("training-code-skill-root:skills")
    if (root / ".claude" / "skills").exists():
        violations.append("duplicate-training-code-skill-root:.claude/skills")

    return sorted(set(violations))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    violations = audit_documents(args.root)
    if violations:
        for violation in violations:
            print(violation)
        return 1
    paths = document_paths(args.root.resolve(strict=True))
    lines = sum(
        len(path.read_text(encoding="utf-8", errors="replace").splitlines()) for path in paths
    )
    print(f"training-code-docs: ok files={len(paths)} lines={lines}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
