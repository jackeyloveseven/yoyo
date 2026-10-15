# Training Code v2 usage

## Install and inspect

```bash
python3 -m pip install -e 'training_code[test]'
mmtrain registry list
```

Importing the core does not load Torch or CUDA. Model and checkpoint dependencies
are optional runtime concerns rather than import-time side effects.

## Validate a run

RunSpecs are immutable, reject unknown fields, and reference registered assets by
ID and SHA-256 rather than accepting filesystem paths or service URLs.

```bash
mmtrain plan validate --spec training_code/recipes/v2/firered_diffusionnft.json
```

Template recipes contain placeholder asset digests and remain disabled. A runtime
agent must resolve registered target, environment, asset, and output IDs before a
qualified launch.

## Execute one phase

Every `phase run` must pass `--expected-sha256` with the canonical SHA-256 of the
spec. `mmtrain plan validate` prints that hash as `spec_sha256`; the CLI re-checks
it before running so the executed plan is byte-identical to the validated one.

```bash
SHA=$(mmtrain plan validate --spec /path/to/run-spec.json | python3 -c "import sys,json;print(json.load(sys.stdin)['spec_sha256'])")
mmtrain phase run --spec /path/to/run-spec.json --expected-sha256 "$SHA" --phase preflight
mmtrain phase run --spec /path/to/run-spec.json --expected-sha256 "$SHA" --phase train
mmtrain phase run --spec /path/to/run-spec.json --expected-sha256 "$SHA" --phase evaluate
mmtrain phase run --spec /path/to/run-spec.json --expected-sha256 "$SHA" --phase publish
```

Only `preflight`, `train`, `evaluate`, and `publish` are valid phases. The engine
owns phase order, construction, events, and termination. Objective plugins own
algorithm-specific training mathematics.

Training Code does not select hosts, poll GPUs, open SSH sessions, manage tmux,
restart failed jobs, or accept free shell text. Those duties belong to the trusted
WebTrainer/Agent control plane.

## Checkpoints and v1 policy import

v2 checkpoints use atomic completion markers and content hashes. Exact resume is
valid only within the same v2 lineage and topology contract.

```bash
mmtrain checkpoint import-policy \
  --source /path/to/v1-checkpoint.pt --source-sha256 <sha256> \
  --source-release training-code-v1.9.0 --model firered --output /path/to/output/
```

The import creates a policy-only lineage receipt. It never imports v1 optimizer,
EMA, history, RNG, or dataset cursor state. Full-state v1 recovery must use the
`training-code-v1.9.0` bundle.

## Qualification

A source-visible recipe is not automatically launchable. Promotion proceeds in
order:

1. CPU contract tests;
2. GPU no-update;
3. one optimizer update;
4. checkpoint readback;
5. frozen evaluation and explicit quality receipt.

Until a runtime receipt is bound, `launch_enabled=false`. Until independent
evaluation is accepted, `promotion_qualified=false`. Current state is read from
WebTrainer and receipts, never from a source-tree status document.
