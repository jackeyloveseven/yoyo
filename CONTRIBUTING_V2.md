# Training Code v2 contribution rules

Run from the repository root:

```bash
PYTHONPATH=training_code/src python3 training_code/check_v2_quality.py
PYTHONPATH=training_code/src:training_code python3 -m pytest -q training_code/tests/v2
python3 training_code/check_docs.py --root training_code
```

The v2 gate enforces:

- core files at most 800 lines, plugins at most 1,200, and no production file above 2,000;
- functions at most 120 lines, eight parameters, and cyclomatic complexity 15;
- no repeated normalized 50-token production window;
- one argparse boundary, no `sys.path` mutation, home path, SSH/process-control import,
  broad fallback, production dependency cycle, or forbidden directory (`scripts/` included);
- active non-test Python at most 103,748 lines, with a non-increasing budget.

Adding a recipe changes data. Adding a model, objective, dataset, or reward changes
a lazy registry entry and plugin, not the engine. Keep algorithm-specific Protocols;
do not introduce a universal training-step abstraction.

A core architecture change above 500 added lines requires an ADR under
[`docs/adr/`](docs/adr/) that names the code it replaces and the deletion delivered
in the same migration series.

Documentation is also gated: no mutable status, handoff, incident timeline, machine
home path, duplicate 50-token fragment or skill source, or broken local link belongs in 2.x. Historical
v1 behavior is linked to an immutable tag instead of copied into current documents.
