# Training Code 1.9

Training Code 1.9 is the final v1 compatibility release. Its source, dependency
baseline, entrypoints, and release manifest are immutable at
[`training-code-v1.9.0`](https://github.com/FlowGPT/mmtrainer/tree/training-code-v1.9.0/training_code).

Use that tag or its release bundle for v1 full-state exact resume. The 2.x branch
does not reproduce v1 launch instructions and does not link historical claims to
the mutable 2.x `release-manifest.json`.

v2 accepts only an explicit policy-only fork from a compatible v1 checkpoint.
It rejects v1 optimizer, EMA, history, RNG, and data-cursor state.
