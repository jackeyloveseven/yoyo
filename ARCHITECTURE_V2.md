# Training Code v2 architecture

Training Code v2 is an installable package with one command boundary and a lazy plugin registry. WebTrainer compiles
an immutable `mmtrainer.run-spec.v2`; `mmtrain` validates its canonical SHA before running a phase. Model paths,
service URLs, hosts and output directories are resolved from registered IDs by the trusted runtime agent and are never
accepted as free component configuration.

```text
WebTrainer recipe -> RunSpec -> mmtrain phase run
                              -> engine lifecycle
                              -> lazy model/objective/dataset/reward plugins
                              -> runtime events/checkpoints
```

## Dependency direction

- `config`, `interfaces`, `registry` and `engine` form the model-neutral core and must import without Torch.
- `nn` provides model-neutral flow interpolation, ODE solvers (`euler_flow_step`, `dpm2_flow_step`), schedules, FSDP2 helpers and `NativeLoRARouter` (sparse LoRA-MoE).
- `algorithms` owns objective math (`sft_dpo`, `diffusion_alignment`, `flow_grpo`, `diffusion_opsd`, `edit_r1`, `flowmimic_opsd`, `reward_matching`, `trajectory_distillation`) and cannot import a concrete model.
- `models`, `datasets` and `rewards` implement registered plugins and composable adapters (`BaseModelAdapter` subclasses); model packages cannot import each other.
- `runtime` owns events, artifacts and checkpoint integrity, not SSH, tmux, GPU reservation or process supervision.
- `examples` demonstrates how to freely compose `nn`, `models`, `datasets`, `algorithms`, `rewards` and `runtime` for SFT, DPO, distillation, RL and custom model onboarding.

The engine deliberately does not provide one universal optimizer step. An ObjectivePlugin owns its inner training
algorithm while the engine owns only phase order, qualification, component construction and structured results.

## Lineage boundary

`training-code-v1.9.0` is the final v1 source contract. Old full-state exact-resume uses its immutable source bundle.
v2 can import only policy tensors through `mmtrain checkpoint import-policy`; the receipt explicitly records that
optimizer, EMA, history, RNG and data cursor were reset.

All built-in v2 catalog entries initially support CPU preflight only. Train/evaluate/publish stay disabled until a
model-specific no-update, one-update and checkpoint-readback receipt is registered.
