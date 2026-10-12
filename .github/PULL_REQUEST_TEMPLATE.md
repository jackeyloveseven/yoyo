## Summary

Briefly describe the component(s), model adapter, or objective added/modified in this PR.

## Architecture & Quality Gate Checklist

- [ ] `PYTHONPATH=src python check_v2_quality.py` passes (no duplicate 50-token windows, cyclomatic complexity <= 15, function args <= 8, no cross-model imports)
- [ ] `python check_structure.py` passes (`mmtrainer.nn` and `mmtrainer.algorithms` remain model-neutral)
- [ ] `python check_docs.py` passes (no broken local links or duplicated documentation fragments)
- [ ] `PYTHONPATH=src:. pytest tests/v2` passes
